import random
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils.Training_utils import *
import losses
from advertorch.context import ctx_noparamgrad_and_eval
from sklearn.preprocessing import StandardScaler
import os

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--gan_loss_type', type=str, default='wgan_gp',
                    help='loss function name. dcgan (default) | hinge | wgan_gp .')
parser.add_argument('--dataset', default='census', type=str, help='Dataset')
parser.add_argument('--model', default='MLP', type=str, help='CNN, MLP')
parser.add_argument('--lr', default=0.01, type=float, help='EDc_lr')
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
        trainargs['num_decoder_feats'] = 30
        trainargs['num_classes'] = 3
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 1e-3
        trainargs['edc_lr'] = 1e-2
        trainargs['da_lr'] = 1e-3
        trainargs['schedule_milestones'] = [3000]
        trainargs['scheduler_gamma'] = 0.1
    elif dataset == 'pedec':
        trainargs['num_decoder_feats'] = 16
        trainargs['num_classes'] = 2
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 5e-5
        trainargs['edc_lr'] = args.lr
        trainargs['da_lr'] = 5e-4
        trainargs['schedule_milestones'] = [100]
        trainargs['scheduler_gamma'] = 0.1
    elif dataset == 'census':
        trainargs['num_decoder_feats'] = 64
        trainargs['num_classes'] = 2
        trainargs['weight_decay'] = 1e-5

        trainargs['e_lr'] = 1e-4
        trainargs['edc_lr'] = 1e-3
        trainargs['da_lr'] = 1e-4
        # Transformer 1,3: 5e-4
        trainargs['schedule_milestones'] = [60]
        trainargs['scheduler_gamma'] = 0.1
    else:
        raise ValueError(f'Dataset not recognized ({dataset})')
    return trainargs


def get_attack(model):
    return LmixPGDAttack_mixed(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                               eps_con=0.2, eps_cat=float(budgets[Dataset]),
                               nb_iter=30, eps_iter_con=0.015,
                               eps_iter_cat=float(budgets[Dataset]) / 15, rand_init=True, clip_min=0.0,
                               clip_max=1.0, targeted=False)


def Training(Dataset, batch_size, n_epoch):
    # devide the dataset into train, validation and test
    trainargs = get_train_args(Dataset)
    device = torch.device("cuda")
    train_idx, test_idx = dataset_split(Dataset)
    X, y = preparation(Dataset)
    X = X.float().numpy()
    y_Train = y[train_idx]
    X_Train = X[train_idx]

    scaler = StandardScaler()
    scaler.fit(X_Train[:, :num_con_feature[Dataset]])
    if not os.path.exists('./dataset/' + Dataset + '_scaler.pkl'):
        pickle.dump(scaler, open('./dataset/' + Dataset + '_scaler.pkl', 'wb'))
    X[:, :num_con_feature[Dataset]] = scaler.transform(X[:, :num_con_feature[Dataset]])
    X = torch.from_numpy(X)
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

    Da_class = SNPDFC3
    if args.model == 'MLP':
        E = MLP_E()
        EDc = MLP_Dc(E).cuda()
        EDa = Da_class(E, num_features=E.hidden_size * 2, num_classes=num_classes[Dataset]).cuda()
    else:
        E = Transformer_E()
        EDc = Transformer_Dc(E).cuda()
        EDa = Da_class(E, num_features=E.hidden_size, num_classes=num_classes[Dataset]).cuda()

    Da_pars, Da_par_names = get_decoder_pars(EDa)

    optDa, optE, optEDc, Dalr_scheduler, Elr_scheduler, EDclr_scheduler = \
        get_optimizers(args,
                       trainargs,
                       {'Da_pars': Da_pars, 'E_pars': E.parameters(), 'EDc_pars': EDc.parameters()}
                       )

    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)

    best_train_cost = 1000000
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
        samples = random.sample(range(n_batches), n_batches)

        adv_train_correct = []
        EDc.train()
        EDa.train()

        # start training with randomly input batches.
        for index in samples:
            rand_it_num = torch.randint(0, 100, (1,))
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            target = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)

            with ctx_noparamgrad_and_eval(EDc):
                adv_data_con, adv_data_cat = adversary.perturb(data[0], data[1], target)
                adv_data_con = adv_data_con.detach()
                adv_data_cat = adv_data_cat.detach()

            optEDc.zero_grad()
            output = EDc(data[0], data[1])
            # errC = F.cross_entropy(output, target, weight=torch.FloatTensor([1, 3]).cuda(), reduction='mean')
            errC = F.cross_entropy(output, target, reduction='mean')
            errC.backward()
            optEDc.step()

            if Dataset == 'census':
                empty_list = E.empty_emb()
                E.embedding.weight.data[empty_list] = 0

            # Train Da for adversary decoding
            optDa.zero_grad()
            output_clean = EDa(data, target)
            output_adv = EDa([adv_data_con, adv_data_cat], target)
            errD = Da_criterion(output_adv.view(-1), output_clean.view(-1))
            if args.gan_loss_type == "wgan_gp":
                gp = losses.gradient_penalty(EDa, E(data[0], data[1]).detach(), E(adv_data_con, adv_data_cat).detach(), device=device)
                errD += 0.1 * gp
            errD.backward()
            optDa.step()

            # Train E for (against) adversary decoding
            optE.zero_grad()
            output_adv = EDa([adv_data_con, adv_data_cat], target)
            with torch.no_grad():
                EDc_output = EDc(adv_data_con, adv_data_cat)
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

            # print(errC.item(), errD.item(), errE.item())
            cost_vector.append(errC.item() + errD.item() + errC.item())
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
        print()

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
    n_batches_IG = int(np.ceil(float(len(X_Test)) / float(512)))
    invalid_sample = invalid_sample_(Dataset)
    EDc.eval()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    # test for the training set
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Train[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = EDc(t_diagnosis_codes[0], t_diagnosis_codes[1])
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

        logit = EDc(t_diagnosis_codes[0], t_diagnosis_codes[1])
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
    X_Test = np.array(X_Test)[:2000]
    y_Test = y_Test[:2000]
    log_attack = open(
        './Logs/' + Dataset + '/training/attack_utils/Attack_%s_Adam.bak' % (Model_Name), 'w+')
    attacker = Attacker_mixed(EDc, log_attack, Dataset)
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
    X_Test = X_Test[:int(0.1 * len(y_Test))]
    y_Test = y_Test[:int(0.1 * len(y_Test))]
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
Model_Type = args.model + '_AFD'
Model_Name = args.model + '_AFD' + '_' + str(args.idx)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch)
print(args)
