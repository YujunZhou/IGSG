from attack_utils.pgd_attack_restart import attack_pgd_restart
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import random
from utils.Training_utils import *
from attack_utils.context import ctx_noparamgrad
from sklearn.preprocessing import StandardScaler
import os

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.008, type=float, help='learning rate')
parser.add_argument('--dataset', default='census', type=str, help='Dataset')
parser.add_argument('--model', default='Transformer', type=str, help='LSTM, MLP')
parser.add_argument('--lmbda', default=0, type=float, help="The parameter lambda for Fast-BAT.")
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
    raise NotImplementedError('The dataset is not implemented.')


def _attack_loss(predictions, labels):
    return -torch.nn.CrossEntropyLoss(reduction='sum')(predictions, labels)


class BatTrainer:
    def __init__(self, args):
        self.args = args
        self.steps = 1
        self.eps_cat = budgets[Dataset]
        self.eps_con = 0.2
        self.attack_lr_cat = self.eps_cat / 4
        self.attack_lr_con = self.eps_con / 4
        if self.args.lmbda != 0.0:
            self.lmbda_cat = self.args.lmbda
        else:
            self.lmbda_cat = 1. / self.attack_lr_cat
            self.lmbda_con = 1. / self.attack_lr_con

        self.constraint_type = 1
        self.log = None
        self.mode = 'fast_bat'

    def test_sa(self, model, data, labels):
        model.eval()
        with torch.no_grad():
            predictions_sa = model(data[0], data[1])
            correct = (torch.argmax(predictions_sa.data, 1) == labels).sum().cpu().item()
        return correct

    def train(self, model, X_Train, y_Train, opt, loss_func, device, scheduler=None, valid_mat=None):

        model.train()
        training_loss = torch.tensor([0.])
        train_sa = torch.tensor([0.])
        train_ra = torch.tensor([0.])

        total = 0

        n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))

        # start training with randomly input batches.
        for index in range(n_batches):
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            labels = y_Train[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)
            real_batch = data[0].shape[0]
            features_con = data[0].shape[1]
            features_cat = data[1].shape[1]
            categories = data[1].shape[2]
            total += real_batch

            # Record SA along with each batch
            train_sa += self.test_sa(model, data, labels)

            model.train()

            if self.mode == "fast_bat":
                z_init_cat = torch.clamp(
                    data[1] + torch.FloatTensor(data[1].shape).uniform_(-self.eps_cat / 20, self.eps_cat / 20).to(
                        device),
                    min=0, max=1
                ) - data[1]
                z_init_cat = z_init_cat * valid_mat
                z_init_cat.requires_grad_(True)
                
                z_init_con = torch.FloatTensor(data[0].shape).uniform_(-self.eps_con, self.eps_con).to(device)
                z_init_con.requires_grad_(True)

                model.clear_grad()
                model.with_grad()
                attack_loss = _attack_loss(model(data[0] + z_init_con, data[1] + z_init_cat), labels)
                grad_attack_loss_delta_cat = \
                    torch.autograd.grad(attack_loss, z_init_cat, retain_graph=True, create_graph=True)[0]
                delta_cat = z_init_cat - self.attack_lr_cat * grad_attack_loss_delta_cat
                delta_cat = torch.clamp(data[1] + delta_cat, min=0, max=1) - data[1]
                delta_cat = delta_cat * valid_mat
                delta_norms_cat = abs(delta_cat.data).sum([1, 2], keepdim=True)
                delta_cat.data = self.eps_cat * delta_cat.data / torch.max(
                    self.eps_cat * torch.ones_like(delta_norms_cat),
                    delta_norms_cat)
                
                grad_attack_loss_delta_con = \
                    torch.autograd.grad(attack_loss, z_init_con, retain_graph=True, create_graph=True)[0]
                delta_con = z_init_con - self.attack_lr_con * grad_attack_loss_delta_con
                delta_con = torch.clamp(delta_con, min=-self.eps_con, max=self.eps_con)
                

                delta_cat = delta_cat.detach().requires_grad_(True)
                delta_con = delta_con.detach().requires_grad_(True)
                attack_loss_second = _attack_loss(model(data[0] + delta_con, data[1] + delta_cat), labels)
                grad_attack_loss_delta_second_cat = \
                    torch.autograd.grad(attack_loss_second, delta_cat, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, features_cat * categories)
                delta_star_cat = delta_cat - self.attack_lr_cat * grad_attack_loss_delta_second_cat.detach().view(
                    data[1].shape)
                # delta_star = torch.clamp(delta_star, min=-self.eps, max=self.eps)
                delta_star_cat = torch.clamp(data[1] + delta_star_cat, min=0, max=1) - data[1]
                delta_star_cat = delta_star_cat * valid_mat
                delta_norms_cat = abs(delta_star_cat.data).sum([1, 2], keepdim=True)
                delta_star_cat.data = self.eps_cat * delta_star_cat.data / torch.max(
                    self.eps_cat * torch.ones_like(delta_norms_cat), delta_norms_cat)
                z_cat = delta_star_cat.clone().detach().view(real_batch, -1)
                
                grad_attack_loss_delta_second_con = \
                    torch.autograd.grad(attack_loss_second, delta_con, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, -1)
                delta_star_con = delta_con - self.attack_lr_con * grad_attack_loss_delta_second_con.detach().view(
                    data[0].shape)
                delta_star_con = torch.clamp(delta_star_con, min=-self.eps_con, max=self.eps_con)
                z_con = delta_star_con.clone().detach().view(real_batch, -1)
                
                # H: (batch, feature * categories)
                z_min_cat = torch.max(-data[1].view(real_batch, -1),
                                      -self.eps_cat * 0.1 * torch.ones_like(data[1].view(real_batch, -1)))
                z_max_cat = torch.min(1 - data[1].view(real_batch, -1),
                                      self.eps_cat * 0.1 * torch.ones_like(data[1].view(real_batch, -1)))
                H_cat = ((z_cat > z_min_cat + 1e-7) & (z_cat < z_max_cat - 1e-7)).to(torch.float32)

                z_min_con = torch.max(-data[0].view(real_batch, -1),
                                      -self.eps_con * torch.ones_like(data[0].view(real_batch, -1)))
                z_max_con = torch.min(1 - data[0].view(real_batch, -1),
                                  self.eps_con * torch.ones_like(data[0].view(real_batch, -1)))
                H_con = ((z_con > z_min_con + 1e-7) & (z_con < z_max_con - 1e-7)).to(torch.float32)

                delta_cur_cat = delta_star_cat.detach().requires_grad_(True)
                delta_cur_con = delta_star_con.detach().requires_grad_(True)

                model.no_grad()
                lgt = model(data[0] + delta_cur_con, data[1] + delta_cur_cat)
                delta_star_loss = loss_func(lgt, labels)
                delta_star_loss.backward()
                delta_outer_grad_cat = delta_cur_cat.grad.view(real_batch, -1)
                delta_outer_grad_con = delta_cur_con.grad.view(real_batch, -1)

                hessian_inv_prod_cat = delta_outer_grad_cat / self.lmbda_cat
                bU_cat = (H_cat * hessian_inv_prod_cat).unsqueeze(-1)
                hessian_inv_prod_con = delta_outer_grad_con / self.lmbda_con
                bU_con = (H_con * hessian_inv_prod_con).unsqueeze(-1)

                model.clear_grad()
                model.with_grad()
                b_dot_product_cat = grad_attack_loss_delta_second_cat.bmm(bU_cat).view(-1).sum(dim=0)
                b_dot_product_cat.backward(retain_graph=True)
                cross_term_cat = [-param.grad / real_batch for param in model.parameters()]
                b_dot_product_con = grad_attack_loss_delta_second_con.bmm(bU_con).view(-1).sum(dim=0)
                b_dot_product_con.backward()
                cross_term_con = [-param.grad / real_batch for param in model.parameters()]

                model.clear_grad()
                model.with_grad()
                predictions = model(data[0] + delta_star_con, data[1] + delta_star_cat)
                logit_clean = model(data[0], data[1])
                train_loss = loss_func(predictions, labels) / real_batch + loss_func(logit_clean, labels) / real_batch * 10
                opt.zero_grad()
                train_loss.backward()

                with torch.no_grad():
                    for p, cross in zip(model.parameters(), cross_term_cat):
                        new_grad = p.grad + cross * 0.1
                        p.grad.copy_(new_grad)
                    for p, cross in zip(model.parameters(), cross_term_con):
                        new_grad = p.grad + cross * 0.1
                        p.grad.copy_(new_grad)

                del cross_term_cat, H_cat, grad_attack_loss_delta_second_cat, b_dot_product_cat,\
                    cross_term_con, H_con, grad_attack_loss_delta_second_con, b_dot_product_con
                opt.step()

            else:
                raise NotImplementedError()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
                if self.log is not None:
                    self.log(model,
                             loss=train_loss.cpu(),
                             accuracy=correct.cpu(),
                             learning_rate=scheduler.get_last_lr()[0],
                             batch_size=real_batch)
            if scheduler:
                scheduler.step()

            training_loss += train_loss.cpu().sum().item()
            train_ra += correct.cpu().sum().item()
        print(train_ra)
        print(training_loss)
        return model, training_loss

    def eval(self, model, X, y, attack_eps, attack_steps, attack_lr, attack_rs, device):
        total = 0
        robust_total = 0
        correct_total = 0
        test_loss = 0

        n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
        samples = random.sample(range(n_batches), n_batches)

        # start training with randomly input batches.
        for index in samples:
            # make X like one hot vectors.
            batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
            labels = y[batch_size * index: batch_size * (index + 1)].cuda()
            data = input_process(batch_diagnosis_codes, Dataset)
            real_batch = data.shape[0]
            total += real_batch

            with ctx_noparamgrad(model):
                perturbed_data = attack_pgd_restart(
                    model=model,
                    X=data,
                    y=labels,
                    eps=attack_eps,
                    alpha=attack_lr,
                    attack_iters=attack_steps,
                    n_restarts=attack_rs,
                    rs=(attack_rs > 1),
                    verbose=False,
                    linf_proj=False,
                    l2_proj=False,
                    l2_grad_update=False,
                    l1_proj=True,
                    cuda=True
                ) + data

            if attack_steps == 0:
                perturbed_data = data

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            predictions = model(perturbed_data)
            robust = torch.argmax(predictions, 1) == labels
            robust_total += robust.sum().cpu().item()

            robust_loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            test_loss += robust_loss.cpu().sum().item()

            if self.log:
                self.log(model=model,
                         accuracy=correct.cpu(),
                         robustness=robust.cpu(),
                         batch_size=real_batch)

        return correct_total, robust_total, total, test_loss / total


def Training(Dataset, batch_size, n_epoch, lr):
    device = torch.device("cuda")
    # devide the dataset into train, validation and test
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
    # train_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3]).cuda())
    train_loss = nn.CrossEntropyLoss()

    valid_mat = valid_mat_(Dataset, model)

    print('training start', file=log_f, flush=True)
    trainer = BatTrainer(args=args)
    model.train()

    best_train_cost = 10000000.0
    epoch_duaration = 0.0
    best_epoch = 0

    training_st = time.time()

    for epoch in range(n_epoch):
        start_time = time.time()
        shuffled_idx = np.arange(len(y_Train))
        np.random.shuffle(shuffled_idx)
        X_Train = X_Train[shuffled_idx]
        y_Train = y_Train[shuffled_idx]
        adjust_learning_rate(args, optimizer, epoch, n_epoch)
        model.train()
        model, train_cost = trainer.train(model=model,
                                          X_Train=X_Train,
                                          y_Train=y_Train,
                                          opt=optimizer,
                                          loss_func=train_loss,
                                          device=device,
                                          valid_mat=valid_mat)

        duration = time.time() - start_time
        train_cost = train_cost[0].item()
        epoch_duaration += duration

        if train_cost < best_train_cost:
            best_train_cost = train_cost
            best_epoch = epoch
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, mean_cost:%f, duration:%f'
              % (epoch, train_cost, duration), file=log_f, flush=True)

        buf = 'Best Epoch:%d, Train_Cost:%f' % (best_epoch, best_train_cost)
        print(buf, file=log_f, flush=True)
        print(train_cost, best_epoch)
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
    X_Test = X_Test.numpy()[:2000]
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


lr = args.lr
Dataset = args.dataset
args.setting = 'FastBAT'

Model_Name = args.model + '_FastBAT' + '_' + str(args.idx)

batch_size = batch_sizes[Dataset]
n_epoch = epochs[Dataset]

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, 100, lr)
print(args)
