import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils.Training_utils import *
import losses
from advertorch.attacks import L1PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--gan_loss_type', type=str, default='wgan_gp',
                      help='loss function name. dcgan (default) | hinge | wgan_gp .')
parser.add_argument('--dataset', default='Splice', type=str, help='Dataset')
parser.add_argument('--model', default='Transformer', type=str, help='CNN, MLP')
parser.add_argument('--lr', default=0.001, type=float, help='EDc_lr')
parser.add_argument('--dis_iters', type=int, default=1)
parser.add_argument('--idx', default=0, type=int, help='running index')
args = parser.parse_args()

# There are two datasets, some models have the same name for the two dataset, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.PEDecModels import *
elif args.dataset == 'census':
    from Models.CensusModels import *
else:
    raise NotImplementedError('Dataset not recognized ({args.dataset})')


def get_optimizers(args, trainargs, model_pars):
    if args.gan_loss_type == "wgan_gp":
        beta1 = 0.0
        beta2 = 0.9
    else:
        beta1 = 0.5
        beta2 = 0.999

    optDa = optim.Adam(model_pars['Da_pars'], lr=trainargs['da_lr'], betas=(beta1, beta2))
    optE = optim.Adam(model_pars['E_pars'], lr=trainargs['e_lr'], betas=(beta1, beta2))
    optEDc = optim.Adam(model_pars['EDc_pars'], lr=trainargs['edc_lr'], betas=(beta1, beta2))
    Dalr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optDa, trainargs['schedule_milestones'],
                                                          gamma=trainargs['scheduler_gamma'],
                                                          last_epoch=-1)
    Elr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optE, trainargs['schedule_milestones'],
                                                         gamma=trainargs['scheduler_gamma'],
                                                         last_epoch=-1)
    EDclr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optEDc, trainargs['schedule_milestones'],
                                                           gamma=trainargs['scheduler_gamma'],
                                                           last_epoch=-1)

    return optDa, optE, optEDc, Dalr_scheduler, Elr_scheduler, EDclr_scheduler


def get_train_args(dataset):
    trainargs = {}
    if dataset == 'Splice':
        trainargs['num_classes'] = 3
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 1e-3
        trainargs['edc_lr'] = 1e-3
        trainargs['da_lr'] = 1e-3
        trainargs['schedule_milestones'] = [300]
        trainargs['scheduler_gamma'] = 0.1
    elif dataset == 'pedec':
        trainargs['num_classes'] = 2
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 1e-4
        trainargs['edc_lr'] = args.lr
        trainargs['da_lr'] = 1e-4
        trainargs['schedule_milestones'] = [100]
        trainargs['scheduler_gamma'] = 0.1
    elif dataset == 'census':
        trainargs['num_classes'] = 2
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 1e-4
        trainargs['edc_lr'] = args.lr
        trainargs['da_lr'] = 1e-4
        trainargs['schedule_milestones'] = [120]
        trainargs['scheduler_gamma'] = 0.1
    else:
        raise ValueError(f'Dataset not recognized ({dataset})')
    return trainargs


def get_attack(model):
    return L1PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=float(budgets[Dataset]),
      nb_iter=20, eps_iter=float(budgets[Dataset])/10, rand_init=True, clip_min=0.0,
      clip_max=1.0, targeted=False)


def Training(Dataset, batch_size, n_epoch):
    # devide the dataset into train, validation and test
    trainargs = get_train_args(Dataset)
    device = torch.device("cuda")
    train_idx, test_idx = dataset_split(Dataset)
    X, y = preparation(Dataset)
    y_Train = y[train_idx]
    X_Train = X[train_idx]
    y_Test = y[test_idx]
    X_Test = X[test_idx]

    output_file = './outputs/' + Dataset + '/' + Model_Name + '/' + str(args.lr) + '/'
    make_dir('./outputs/')
    make_dir('./outputs/' + Dataset + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/' + str(args.lr) + '/')
    make_dir('./Logs/')
    make_dir('./Logs/' + Dataset)
    make_dir('./Logs/' + Dataset + '/training/')
    make_dir('./Logs/' + Dataset + '/training/details/')
    make_dir('./Logs/' + Dataset + '/training/attack_utils/')

    log_f = open(
        './Logs/' + Dataset + '/training/details/TEST_%s.bak' % (
            Model_Name), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    if args.model == 'MLP':
        E, Dc = MLP_E(), MLP_Dc()
    elif args.model == 'Transformer':
        E, Dc = Transformer_E(), Transformer_Dc()
    EDc = nn.Sequential(E, Dc).cuda()
    Da_class = SNPDFC3

    EDa = Da_class(E, num_features=E.hidden_size, num_classes=num_classes[Dataset]).cuda()
    Da_pars, Da_par_names = get_decoder_pars(EDa)

    optDa, optE, optEDc, Dalr_scheduler, Elr_scheduler, EDclr_scheduler = \
        get_optimizers(args,
                       trainargs,
                       {'Da_pars': Da_pars, 'E_pars': E.parameters(), 'EDc_pars': EDc.parameters()}
                       )

    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)

    best_train_cost = 10000000.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    adversary = get_attack(EDc)
    E_criterion = losses.GenLoss(args.gan_loss_type)
    Da_criterion = losses.DisLoss(args.gan_loss_type)

    training_st = time.time()

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        shuffled_idx = np.arange(len(y_Train))
        np.random.shuffle(shuffled_idx)
        X_Train = X_Train[shuffled_idx]
        y_Train = y_Train[shuffled_idx]

        adv_train_correct = []
        EDc.train()
        EDa.train()

        # start training with randomly input batches.
        for index in range(n_batches):
            rand_it_num = torch.randint(0, 100, (1,))
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            target = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)

            with ctx_noparamgrad_and_eval(EDc):
                adv_data = adversary.perturb(data, target).detach()

            optEDc.zero_grad()
            output = EDc(data)
            errC = F.cross_entropy(output, target, reduction='mean')
            errC.backward()
            optEDc.step()

            if Dataset == 'census':
                empty_list = E.empty_emb()
                E.embedding.weight.data[empty_list] = 0

            # Train Da for adversary decoding
            optDa.zero_grad()
            output_clean = EDa(data, target)
            output_adv = EDa(adv_data, target)
            errD = Da_criterion(output_adv.view(-1), output_clean.view(-1))
            if args.gan_loss_type == "wgan_gp":
                gp = losses.gradient_penalty(EDa, E(data).detach(), E(adv_data).detach(), device=device)
                errD += 0.1 * gp
            errD.backward()
            optDa.step()

            # Train E for (against) adversary decoding
            optE.zero_grad()
            output_adv = EDa(adv_data, target)
            with torch.no_grad():
                EDc_output = EDc(adv_data)
            EDc_train_pred = EDc_output.max(1, keepdim=True)[1]
            adv_train_correct.append(
                100. * EDc_train_pred.eq(target.view_as(EDc_train_pred)).sum().item() / float(len(EDc_train_pred)))
            errE = E_criterion(output_adv.view(-1), None)
            if rand_it_num % args.dis_iters == 0:
                errE.backward()
                optE.step()

            if Dataset == 'census':
                empty_list = E.empty_emb()
                E.embedding.weight.data[empty_list] = 0

            cost_vector.append(errC.item()+errD.item()+errC.item())
            # print(errC.item(), errD.item(), errE.item())
            iteration += 1

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        epoch_duaration += duration

        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        if train_cost < best_train_cost:
            best_train_cost = train_cost
            best_epoch = epoch
            torch.save(EDc.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)

        buf = 'Best Epoch:%d, Train_Cost:%f' % (best_epoch, best_train_cost)
        print(buf, file=log_f, flush=True)

        Dalr_scheduler.step()
        Elr_scheduler.step()
        EDclr_scheduler.step()

    training_st = time.time() - training_st
    store_time(Dataset, Model_Name, training_st)

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
    print(best_parameters_file)
    EDc.load_state_dict(torch.load(best_parameters_file))
    torch.save(EDc.state_dict(), './classifier/' + Dataset + '_' + Model_Name + '.par',
               _use_new_zipfile_serialization=False)
    EDc.eval()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    # test for the training set
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Train[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = EDc(t_diagnosis_codes)
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    log_a = open(
        './Logs/' + Dataset + '/training/TEST____%s_Adam.bak' % (Model_Name), 'w+')
    print(best_parameters_file, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    # test for test set
    y_true = np.array([])
    y_pred = np.array([])
    n_batches_test = int(np.ceil(float(len(X_Test)) / float(batch_size)))
    for index in range(n_batches_test):  # n_batches

        batch_diagnosis_codes = X_Test[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Test[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = EDc(t_diagnosis_codes)
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    ### OMPGS attack_utils
    shuffle_index = np.arange(len(y_Test))
    np.random.seed(1)
    np.random.shuffle(shuffle_index)
    X_Test = X_Test[shuffle_index]
    y_Test = y_Test[shuffle_index]
    X_Test = np.array(X_Test)[:500]
    y_Test = y_Test[:500]
    log_attack = open(
        './Logs/' + Dataset + '/training/attack_utils/Attack_%s_Adam.bak' % (Model_Name), 'w+')
    attacker = Attacker(EDc, log_attack, Dataset)
    success = 0
    pred_right = 0
    for i in range(len(y_Test)):
        print(i)
        print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

        sample = X_Test[i]
        label = int(y_Test[i])

        g_process, mf_process, greedy_set_process, changed_set_process, query_num, robust_flag, \
        greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, \
        num_changed, flip_funccall, flip_set, iteration = attacker.attack(sample, label)

        success += (robust_flag == 1)
        pred_right += (robust_flag != -1)
    print('OMPGS adv acc:', success / len(y_Test), file=log_a, flush=True)
    print('OMPGS adv acc:', success / len(y_Test))

    # FSGS attack_utils
    success = 0
    pred_right = 0
    X_Test = np.array(X_Test)[:101]
    y_Test = y_Test[:101]
    for i in range(len(y_Test)):
        print(i)
        print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

        sample = X_Test[i]
        label = int(y_Test[i])

        g_process, mf_process, greedy_set_process, changed_set_process, query_num, robust_flag, \
        greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, \
        num_changed, flip_funccall, flip_set, iteration = attacker.attack_FSGS(sample, label)

        success += (robust_flag == 1)
        pred_right += (robust_flag != -1)
    print('FSGS adv acc:', success / len(y_Test), file=log_a, flush=True)
    print('FSGS adv acc:', success / len(y_Test))


Dataset = args.dataset
args.setting = 'AFD'

Model_Name = args.model + '_AFD' + '_' + str(args.idx)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch)
print(args)
