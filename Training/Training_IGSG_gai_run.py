from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils.Training_utils import *
# from tensorboardX import SummaryWriter

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--dataset', default='Splice', type=str, help='Dataset (Splice, PEDec)')
parser.add_argument('--setting', default='IGSG', type=str, help='Normal, IGSG, IG, SG, IGR, JR, IGSG_VG, IGSG_VSG, SGSG, IGIG')
parser.add_argument('--model', default='MLP', type=str, help='MLP, CNN')
parser.add_argument('--alpha', default=0, type=float, help='setting of alpha')
parser.add_argument('--beta', default=0, type=float, help='setting of beta')
parser.add_argument('--idx', default=0, type=int, help='running index')
parser.add_argument('--T', default=20, type=int, help='setting of T')
parser.add_argument('--R', default=5, type=int, help='setting of R')
parser.add_argument('--gamma', default=0, type=float, help='setting of gamma')
args = parser.parse_args()

# There are two datasets, some models have the same name for the two dataset, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.PEDecModels import *
elif args.dataset == 'census':
    from Models.CensusModels import *
else:
    raise NotImplementedError(f'Dataset not recognized ({args.dataset})')


def Val_Evaluate(model, Dataset, batch_size, X_val, y_val, valid_mat, invalid_sample):
    model.eval()
    n_batches = int(np.ceil(float(len(X_val)) / float(batch_size)))
    CEloss = torch.nn.CrossEntropyLoss().cuda()
    loss_sum = 0
    pred_right = 0
    clean_loss_sum = 0
    grad_loss_sum = 0
    attr_loss_sum = 0
    for index in range(n_batches):
        grad_loss = torch.tensor(0)
        attr_loss = torch.tensor(0)
        batch_diagnosis_codes = X_val[batch_size * index: batch_size * (index + 1)]
        t_labels = y_val[batch_size * index: batch_size * (index + 1)].cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

        if 'IG' in args.setting and args.setting != 'IGR':
            batch_attribution, IG_matrix_all = IntegratedGradient_Batch(batch_diagnosis_codes, t_labels, Dataset,
                                                                        invalid_sample, model, steps=args.T)
            mean_attribution = torch.mean(batch_attribution, dim=0)
            attr_loss = torch.norm(mean_attribution[:-1] - mean_attribution[1:], p=1) * beta

        if 'SG' in args.setting:
            batch_diagnosis_codes_sm = smooth_sampling(batch_diagnosis_codes, Dataset, sm_num)
            t_labels_sm = y_val[batch_size * index: batch_size * (index + 1)].repeat(sm_num).cuda()
            t_diagnosis_codes_sm = input_process(batch_diagnosis_codes_sm, Dataset)
            t_diagnosis_codes_sm = torch.autograd.Variable(t_diagnosis_codes_sm.data, requires_grad=True)
            logit = model(t_diagnosis_codes_sm)
            loss = CEloss(logit, t_labels_sm)
            loss.backward(retain_graph=True)
            grad_sm = t_diagnosis_codes_sm.grad.data
            if args.setting == 'SGSG':
                attributions = torch.norm(grad_sm, p=2, dim=[2])
                mean_attribution = torch.mean(attributions, dim=0)
                attr_loss = torch.norm(mean_attribution[:-1] - mean_attribution[1:], p=1) * delta
            if Dataset_type[Dataset] == 'multi' and 'VSG' not in args.setting:
                grad_orig_cate = torch.sum(grad_sm * t_diagnosis_codes_sm, dim=2, keepdim=True).repeat(1, 1,
                                                                                                       num_category[
                                                                                                           Dataset])
                grad_orig_cate = grad_orig_cate * valid_mat
                grad_sm = grad_sm - grad_orig_cate
            grad_norm = torch.norm(grad_sm) / np.sqrt(sm_num)
            grad_loss = grad_norm * alpha

        elif args.setting == 'IGSG_VG':
            logit = model(t_diagnosis_codes)
            loss = CEloss(logit, t_labels)
            loss.backward(retain_graph=True)
            grad = t_diagnosis_codes.grad.data
            if Dataset_type[Dataset] == 'multi':
                grad_orig_cate = torch.sum(grad * t_diagnosis_codes, dim=2, keepdim=True).repeat(1, 1,
                                                                                                 num_category[
                                                                                                     Dataset])
                grad_orig_cate = grad_orig_cate * valid_mat
                grad = grad - grad_orig_cate
            grad_norm = torch.norm(grad)
            grad_loss = grad_norm * alpha

        elif args.setting == 'IGIG':
            grad_norm = torch.norm(IG_matrix_all)
            grad_loss = grad_norm * epsilon

        logit = model(t_diagnosis_codes)
        clean_loss = CEloss(logit, t_labels)
        loss = clean_loss + grad_loss + attr_loss
        if args.setting == 'JR':
            R = reg(t_diagnosis_codes, logit)
            jacobian_loss = R * gamma
            loss = clean_loss + jacobian_loss
        if args.setting == 'IGR':
            logit = model(t_diagnosis_codes)
            loss = CEloss(logit, t_labels)
            loss.backward(retain_graph=True)
            grad = t_diagnosis_codes.grad
            grad_norm = torch.norm(grad)
            grad_loss = grad_norm * alpha
            loss = clean_loss + grad_loss
        loss_sum += loss.item()
        clean_loss_sum += clean_loss.item()
        grad_loss_sum += grad_loss.item()
        attr_loss_sum += attr_loss.item()
        pred_right += torch.sum(torch.argmax(logit, dim=1) == t_labels).item()
    print('Val Acc:', pred_right / len(X_val), 'Clean Loss:', clean_loss_sum / len(X_val), 'Grad Loss:', grad_loss_sum / len(X_val), 'Attr Loss:', attr_loss_sum / len(X_val))
    return loss_sum

def Training(Dataset, batch_size, n_epoch, lr):
    train_idx, test_idx = dataset_split(Dataset)
    X, y = preparation(Dataset)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
    y_Train = y[train_idx]
    X_Train = X[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
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
    make_dir('./Logs/' + Dataset + '/' + Model_Type + '/')

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

    CEloss = torch.nn.CrossEntropyLoss().cuda()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    # defines the invalid sample x' using in Integrated Gradient
    invalid_sample = invalid_sample_(Dataset)
    print('training start', file=log_f, flush=True)
    model.train()

    best_val_loss = 1000000.0
    best_train_loss = 1000000.0
    epoch_duaration = 0.0
    best_epoch = 0
    attributions = torch.tensor([])
    IG_matrix_all = torch.tensor([1 / num_feature[Dataset]] * num_feature[Dataset])
    # find the valid categories for each feature
    valid_mat = valid_mat_(Dataset, model)

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
        clean_loss_sum = 0
        grad_loss_sum = 0
        attr_loss_sum = 0
        model.train()
        train_loss = 0
        for index in range(n_batches):
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            t_labels = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
            grad_loss = torch.tensor(0)
            attr_loss = torch.tensor(0)

            if 'IG' in args.setting and args.setting != 'IGR' and epoch % 10 == 0:
                batch_attribution, IG_matrix_all = IntegratedGradient_Batch(batch_diagnosis_codes, t_labels, Dataset,
                                                                            invalid_sample, model)
                mean_attribution = torch.mean(batch_attribution, dim=0)
                attr_loss = torch.norm(mean_attribution[:-1] - mean_attribution[1:], p=1) * beta
                if epoch % 100 == 0 and index == 0:
                    print(IG_matrix_all)
                    print(batch_attribution)

            if 'SG' in args.setting:
                batch_diagnosis_codes_sm = smooth_sampling(batch_diagnosis_codes, Dataset, sm_num)
                t_labels_sm = y_Train[batch_size * index: batch_size * (index + 1)].repeat(sm_num).cuda()
                t_diagnosis_codes_sm = input_process(batch_diagnosis_codes_sm, Dataset)
                t_diagnosis_codes_sm = torch.autograd.Variable(t_diagnosis_codes_sm.data, requires_grad=True)
                logit = model(t_diagnosis_codes_sm)
                loss = CEloss(logit, t_labels_sm)
                # grad_sm = torch.autograd.grad(loss, t_diagnosis_codes_sm, create_graph=True)[0]
                loss.backward(retain_graph=True)
                grad_sm = t_diagnosis_codes_sm.grad.data
                if args.setting == 'SGSG':
                    attributions = torch.norm(grad_sm, p=2, dim=[2])
                    mean_attribution = torch.mean(attributions, dim=0)
                    attr_loss = torch.norm(mean_attribution[:-1] - mean_attribution[1:], p=1) * delta
                if Dataset_type[Dataset] == 'multi' and 'VSG' not in args.setting:
                    grad_orig_cate = torch.sum(grad_sm * t_diagnosis_codes_sm, dim=2, keepdim=True).repeat(1, 1,
                                                                                                           num_category[
                                                                                                               Dataset])
                    grad_orig_cate = grad_orig_cate * valid_mat
                    grad_sm = grad_sm - grad_orig_cate
                grad_norm = torch.norm(grad_sm) / np.sqrt(sm_num)
                grad_loss = grad_norm * alpha
            elif args.setting == 'IGSG_VG':
                logit = model(t_diagnosis_codes)
                loss = CEloss(logit, t_labels)
                loss.backward(retain_graph=True)
                grad = t_diagnosis_codes.grad.data
                if Dataset_type[Dataset] == 'multi':
                    grad_orig_cate = torch.sum(grad * t_diagnosis_codes, dim=2, keepdim=True).repeat(1, 1,
                                                                                                     num_category[
                                                                                                         Dataset])
                    grad_orig_cate = grad_orig_cate * valid_mat
                    grad = grad - grad_orig_cate
                grad_norm = torch.norm(grad)
                grad_loss = grad_norm * alpha

            elif args.setting == 'IGIG' and epoch % 10 == 0:
                grad_norm = torch.norm(IG_matrix_all)
                grad_loss = grad_norm * epsilon

            optimizer.zero_grad()

            logit = model(t_diagnosis_codes)
            clean_loss = CEloss(logit, t_labels)

            loss = clean_loss + grad_loss + attr_loss
            if args.setting == 'JR':
                R = reg(t_diagnosis_codes, logit)
                jacobian_loss = R * gamma
                loss = clean_loss + jacobian_loss
            if args.setting == 'IGR':
                logit = model(t_diagnosis_codes)
                loss = CEloss(logit, t_labels)
                loss.backward(retain_graph=True)
                grad = t_diagnosis_codes.grad
                grad_norm = torch.norm(grad)
                grad_loss = grad_norm * alpha
                loss = clean_loss + grad_loss
            clean_loss_sum += clean_loss.item()
            grad_loss_sum += grad_loss.item()
            attr_loss_sum += attr_loss.item()
            loss.backward()

            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            iteration += 1
            train_loss += loss.item()
        if epoch % 10 == 0:
            print('epoch:', epoch)
            print('attributions:', attributions)
            print('clean_loss:', clean_loss_sum, 'grad_loss:', grad_loss_sum, 'attr_loss:', attr_loss_sum)
            print('epoch:', epoch, file=log_f, flush=True)
            print('attributions:', attributions, file=log_f, flush=True)
            print('clean_loss:', clean_loss_sum, 'grad_loss:', grad_loss_sum, 'attr_loss:', attr_loss_sum, file=log_f, flush=True)

        duration = time.time() - start_time
        epoch_duaration += duration

        val_loss = Val_Evaluate(model, Dataset, batch_size, X_val, y_val, valid_mat, invalid_sample)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, val_loss:%f, duration:%f'
              % (epoch, val_loss, duration), file=log_f, flush=True)

        if epoch / n_epoch < 0.6 and (epoch+1) / n_epoch >= 0.6:
            if train_loss > 1.2 * best_train_loss:
                best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
                model.load_state_dict(torch.load(best_parameters_file))
        elif epoch / n_epoch < 0.75 and (epoch+1) / n_epoch >= 0.85:
            if train_loss > 1.2 * best_train_loss:
                best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
                model.load_state_dict(torch.load(best_parameters_file))


    # training_st = time.time() - training_st
    # store_time(Dataset, Model_Name, training_st)

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
    print(best_parameters_file)
    model.load_state_dict(torch.load(best_parameters_file))
    torch.save(model.state_dict(), './classifier/' + Dataset + '_' + Model_Name + '.par',
               _use_new_zipfile_serialization=False)
    n_batches_IG = int(np.ceil(float(len(X_Test)) / float(512)))
    best_attributions, _ = IntegratedGradient(X_Test, y_Test, Dataset, n_batches_IG, invalid_sample, model)
    total_variation_loss = torch.norm(best_attributions[:-1] - best_attributions[1:], p=1)
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

    accuacy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuacy, precision, recall, f1))

    log_a = open(
        './Logs/' + Dataset + '/training/TEST____%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    print(best_parameters_file, file=log_a, flush=True)
    print('alpha, beta:', alpha, beta, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuacy, precision, recall, f1), file=log_a, flush=True)

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

    accuacy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuacy, precision, recall, f1))

    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuacy, precision, recall, f1), file=log_a, flush=True)

    print(best_attributions, file=log_a, flush=True)
    print('total variation loss:', total_variation_loss, file=log_a, flush=True)

    ### OMPGS attack_utils
    X_Test = np.array(X_Test)[:500]
    y_Test = y_Test[:500]
    log_attack = open(
        './Logs/' + Dataset + '/training/attack_utils/Attack_%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    attacker = Attacker(model, log_attack, Dataset)
    success = 0
    pred_right = 0
    for i in range(len(y_Test)):
        # print(i)
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
        # print(i)
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
sm_num = args.R
Model_Type = args.model + '_' + args.setting
Model_Name = args.model + '_' + args.setting + '_' + str(args.idx) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' + str(args.T) + '_' + str(args.R)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]
if args.alpha == 0:
    alpha = alphas[Dataset]
else:
    alpha = args.alpha
if args.beta == 0:
    beta = betas[Dataset]
else:
    beta = args.beta
if args.gamma == 0:
    gamma = gammas[Dataset]
else:
    gamma = args.gamma
delta = deltas[Dataset]
epsilon = epsilons[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if args.setting == 'JR':
    from jacobian import JacobianReg
    reg = JacobianReg()

Training(Dataset, batch_size, n_epoch, lr)
print(args)
