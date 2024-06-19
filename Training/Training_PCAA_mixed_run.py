from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils.Training_utils import *
from sklearn.preprocessing import StandardScaler
import os
from Attacks.PCAA_mixed import Attacker

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--dataset', default='census', type=str, help='Dataset')
parser.add_argument('--model', default='Transformer', type=str, help='LSTM, MLP')
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
    raise NotImplementedError('Dataset not implemented')

lrs = {'census': 0.9}

def Training(Dataset, batch_size, n_epoch, lr):
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
    # define cross entropy loss function
    CEloss = torch.nn.CrossEntropyLoss().cuda()
    # CEloss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3])).cuda()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)
    model.train()

    valid_mat = valid_mat_(Dataset, model)

    best_train_cost = 1000000.0
    epoch_duaration = 0.0
    best_epoch = 0

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

            with ctx_noparamgrad_and_eval(model):
                attacker = Attacker('', '', Dataset, Model_Type='', itermax=7, lr=lrs[Dataset], model=model)
                adv_data_con, adv_data_cat = attacker.generate(batch_diagnosis_codes, t_labels, 5, eval_num=10)
                adv_data_con = adv_data_con.detach()
                adv_data_cat = one_hot_samples(adv_data_cat, Dataset).cuda().detach()
                model.train()
                model.apply(tune_bn)
                for p in model.parameters():
                    p.requires_grad = True

            adv_data_cat = adv_data_cat * valid_mat
            optimizer.zero_grad()
            logit_adv = model(adv_data_con, adv_data_cat)
            logit = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
            loss = CEloss(logit_adv, t_labels) + CEloss(logit, t_labels) *10
            loss.backward()

            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            iteration += 1

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        epoch_duaration += duration

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

        if epoch / n_epoch < 0.6 and (epoch+1) / n_epoch >= 0.6:
            if train_cost > 1.2 * best_train_cost:
                best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
                model.load_state_dict(torch.load(best_parameters_file))
        elif epoch / n_epoch < 0.85 and (epoch+1) / n_epoch >= 0.85:
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

        logit = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
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

        logit = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
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
        './Logs/' + Dataset + '/training/attack_utils/Attack_%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    attacker = Attacker_mixed(model, log_attack, Dataset)
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
args.setting = 'PCAA'
Model_Type = args.model + '_PCAA'
Model_Name = args.model + '_PCAA' + '_' + str(args.idx)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(args)
Training(Dataset, batch_size, n_epoch, lr)

