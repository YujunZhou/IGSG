from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils.Training_utils import *
from torch.autograd import Variable
import torch.optim as optim

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.07, type=float, help='learning rate')
parser.add_argument('--dataset', default='Splice', type=str, help='Dataset')
parser.add_argument('--model', default='MLP', type=str, help='LSTM, MLP')
parser.add_argument('--beta', default=1, type=float, help='1/lambda')
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
    raise NotImplementedError('The dataset is not implemented yet.')


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                epsilon=budgets[args.dataset],
                perturb_steps=15,
                beta=args.beta,
                ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example with l1 norm
    delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    delta = Variable(delta.data, requires_grad=True)

    # Setup optimizers
    optimizer_delta = optim.Adam([delta], lr=epsilon / perturb_steps * 2)

    valid_mat = valid_mat_(Dataset, model)

    for _ in range(perturb_steps):
        adv = x_natural + delta

        # optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
        loss.backward()
        # renorming gradient
        grad_norms = delta.grad.view(batch_size, -1).norm(p=1, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        optimizer_delta.step()

        # projection
        delta.data = delta.data * valid_mat
        delta.data.add_(x_natural)
        delta.data.clamp_(0, 1).sub_(x_natural)
        delta.data.renorm_(p=1, dim=0, maxnorm=epsilon)
    x_adv = Variable(x_natural + delta, requires_grad=False)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss



def Training(Dataset, batch_size, n_epoch, lr):
    train_idx, test_idx = dataset_split(Dataset)
    X, y = preparation(Dataset)
    y_Train = y[train_idx]
    X_Train = X[train_idx]
    y_Test = y[test_idx]
    X_Test = X[test_idx]

    output_file = './outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/'
    make_dir('./outputs/')
    make_dir('./outputs/' + Dataset + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/')
    make_dir('./Logs/')
    make_dir('./Logs/' + Dataset)
    make_dir('./Logs/' + Dataset + '/training/')
    make_dir('./Logs/' + Dataset + '/training/details/')
    make_dir('./Logs/' + Dataset + '/training/attack_utils/')

    log_f = open(
        './Logs/' + Dataset + '/training/details/TEST_%s_%s.bak' % (
            Model_Name, lr), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    if args.model == 'Transformer':
        model = Transformer()
    else:
        model = MLP()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    print('done!', file=log_f, flush=True)
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)
    model.train()

    best_train_cost = 100000
    epoch_duaration = 0.0
    best_epoch = 0.0

    training_st = time.time()

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        shuffled_idx = np.arange(len(y_Train))
        np.random.shuffle(shuffled_idx)
        X_Train = X_Train[shuffled_idx]
        y_Train = y_Train[shuffled_idx]
        adjust_learning_rate(args, optimizer, epoch, n_epoch)

        # start training with randomly input batches.
        for index in range(n_batches):
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            t_labels = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

            optimizer.zero_grad()
            loss = trades_loss(model, t_diagnosis_codes, t_labels, optimizer, epsilon=float(budgets[Dataset]))
            loss.backward()

            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            iteration += 1

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        epoch_duaration += duration

        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        if train_cost < best_train_cost:
            best_train_cost = train_cost
            best_epoch = epoch
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, mean_cost:%f, duration:%f'
              % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        buf = 'Best Epoch:%d, Train_Cost:%f' % (best_epoch, best_train_cost)
        print(buf, file=log_f, flush=True)
        print()

        if epoch / n_epoch < 0.6 and (epoch + 1) / n_epoch >= 0.6:
            if train_cost > 1.2 * best_train_cost:
                best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
                model.load_state_dict(torch.load(best_parameters_file))
        elif epoch / n_epoch < 0.85 and (epoch + 1) / n_epoch >= 0.85:
            if train_cost > 1.2 * best_train_cost:
                best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
                model.load_state_dict(torch.load(best_parameters_file))

    training_st = time.time() - training_st
    store_time(Dataset, Model_Name, training_st)

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
    print(best_parameters_file)
    model.load_state_dict(torch.load(best_parameters_file))
    torch.save(model.state_dict(), './classifier/' + Dataset + '_' + Model_Name + '.par',
               _use_new_zipfile_serialization=False)
    model.eval()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    # test for the training set
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Train[batch_size * index: batch_size * (index + 1)].numpy()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        logit = model(t_diagnosis_codes)
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
        './Logs/' + Dataset + '/training/TEST____%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
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

        logit = model(t_diagnosis_codes)
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
        './Logs/' + Dataset + '/training/attack_utils/Attack_%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    attacker = Attacker(model, log_attack, Dataset)
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


lr = args.lr
Dataset = args.dataset
args.setting = 'TRADES'
Model_Type = args.model + '_TRADES'
Model_Name = args.model + '_TRADES' + '_' + str(args.idx)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch, lr)
print(args)
