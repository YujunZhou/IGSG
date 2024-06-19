import time
import argparse
import torch.nn.functional as F
from filelock import FileLock
from utils.utils import *

lrs = {'Splice': 0.1, 'pedec': 0.03, 'census': 0.22}
itermaxs = {'Splice': 500, 'pedec': 100, 'census': 50}
taus = {'Splice': 1, 'pedec': 1.25, 'census': 2}
omegas = {'Splice': 1, 'pedec': 100, 'census': 1}

parser = argparse.ArgumentParser(description='PCAA')
# parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='Splice', type=str, help='dataset')
parser.add_argument('--modeltype', default='MLP_Normal', type=str, help='model type')
# parser.add_argument('--time', default=30, type=int, help='time limit')
parser.add_argument('--t', default='True', type=str, help='test set or whole set')
args = parser.parse_args()
def Expect_GumbelSM_grad(model, prob, inputs, label, alpha, epsilon, Dataset, iters=500):  ### modified
    model.train()
    model.apply(fix_bn)
    celoss = nn.CrossEntropyLoss()

    prob2 = prob.repeat(iters, 1, 1)
    z = nn.functional.gumbel_softmax(prob2, dim=2, tau=taus[Dataset], hard=False)
    # tau: Splice 1 PEDec: 2

    label_tensor = torch.tensor([label] * iters).long().cuda()
    ce = celoss(model(z), label_tensor)

    ## penalized distance
    iter_inputs = inputs.repeat(iters, 1, 1)
    dist = torch.sum(-torch.log(z + 0.001) * iter_inputs, dim=2)  ## cross entropy loss
    # print('dist.shape', dist.shape)
    # input(123)
    dist = torch.mean(F.relu(torch.mean(dist, dim=1) - epsilon))  ## only penalize over bar = 1.5

    loss = ce - alpha * dist
    grad = torch.autograd.grad(loss, prob)[0]
    model.eval()

    # print('CE Loss', ce.item(), 'Dist', dist.item(), 'Gradient Norm', torch.sum(torch.abs(grad)).item())
    return grad


def Expect_GumbelSM_grad_batch(model, prob, inputs, labels, alpha, epsilon, Dataset, iters=10):  ### modified

    model.train()
    model.apply(fix_bn)
    celoss = nn.CrossEntropyLoss()

    prob2 = prob.repeat(iters, 1, 1, 1)
    z = nn.functional.gumbel_softmax(prob2, dim=3, tau=taus[Dataset], hard=False)
    z = z.reshape(-1, z.shape[-2], z.shape[-1])
    label_tensor = labels.repeat(iters)
    ce = celoss(model(z), label_tensor)

    ## penalized distance
    iter_inputs = inputs.repeat(iters, 1, 1, 1).reshape(-1, inputs.shape[-2], inputs.shape[-1])
    dist = torch.sum(-torch.log(z + 0.001) * iter_inputs, dim=2)  ## cross entropy loss
    # print('dist.shape', dist.shape)
    # input(123)
    dist = torch.mean(F.relu(torch.mean(dist, dim=1) - epsilon))  ## only penalize over bar = 1.5

    loss = ce - alpha * dist
    grad = torch.autograd.grad(loss, prob)[0]
    model.eval()

    # print('CE Loss', ce.item(), 'Dist', dist.item(), 'Gradient Norm', torch.sum(torch.abs(grad)).item())
    return grad


class Attacker(object):
    def __init__(self, best_parameters_file, log_f, Dataset, Model_Type, itermax=7, lr=0.1, model=None):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        if Dataset == 'Splice':
            from Models.SpliceModels import MLP, MLP_E, MLP_Dc, Transformer, Transformer_E, Transformer_Dc
        elif Dataset == 'pedec':
            from Models.PEDecModels import MLP, MLP_E, MLP_Dc, Transformer, Transformer_E, Transformer_Dc
        if best_parameters_file != '':
            # load the model
            if 'Transformer' in Model_Type:
                if 'AFD' in Model_Type:
                    E, Dc = Transformer_E(), Transformer_Dc()
                    self.model = nn.Sequential(E, Dc)
                else:
                    self.model = Transformer()
            elif 'CNN' in Model_Type:
                if 'AFD' in Model_Type:
                    E, Dc = CNN_E(), CNN_Dc()
                    self.model = nn.Sequential(E, Dc)
                else:
                    self.model = CNN()
            elif 'LSTM' in Model_Type:
                if 'AFD' in Model_Type:
                    E, Dc = LSTM_E(), LSTM_Dc()
                    self.model = nn.Sequential(E, Dc)
                else:
                    self.model = LSTM()
            else:
                if 'AFD' in Model_Type:
                    E, Dc = MLP_E(), MLP_Dc()
                    self.model = nn.Sequential(E, Dc)
                else:
                    self.model = MLP()
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            # load the trained parameters of the classifier
            self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        else:
            self.model = model
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        self.itermax = itermax
        self.tau = taus[Dataset]
        self.lr = lr
        self.Dataset = Dataset

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.LongTensor(np.array([funccall]))
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, self.Dataset)
        return t_diagnosis_codes

    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall != new_funccall)[0])
        return diff_set

    def attack(self, funccall, y, budget, eval_num, alpha=10, epsilon=0.1):
        print()
        st = time.time()
        one_hot_sample = self.input_handle(funccall)
        none_ground_truth = list(range(self.n_labels))
        none_ground_truth.remove(y)
        logit = self.model(one_hot_sample)
        if 'Transformer' in args.modeltype:
            logit = torch.softmax(logit, dim=-1)
        pred = torch.argmax(logit).item()
        logit = logit.cpu().detach().numpy()
        none_ground_truth_highest_value = np.max(logit[0, none_ground_truth])
        y_value = logit[0, y]
        mf = none_ground_truth_highest_value - y_value
        if pred != y:
            # print('Wrong classification originally', file=self.log_f, flush=True)
            print('Wrong classification originally')
            return -1, [], mf
        self.prob = torch.clone(one_hot_sample) * omegas[self.Dataset]
        self.prob.requires_grad = True
        for k in range(self.itermax):
            if k % 50 == 0:
                print('pgd step ' + str(k))
            grad = Expect_GumbelSM_grad(self.model, self.prob, one_hot_sample, y, alpha, epsilon, self.Dataset)

            self.prob = self.prob + self.lr * torch.sign(grad)
            self.prob = torch.clip(self.prob, min=1e-3, max=15)
            self.prob.detach()
            self.prob.requires_grad_
        self.model.eval()
        prob3 = self.prob.repeat(eval_num, 1, 1, 1)
        z = nn.functional.gumbel_softmax(prob3, dim=3, tau=taus[self.Dataset], hard=False)

        changed_nodes = []
        outputs = []
        succ_rate = 0


        for j in range(eval_num):
            dist = torch.sum(torch.abs(z[j] - one_hot_sample)) / 2
            logit = self.model(z[j])
            if 'Transformer' in args.modeltype:
                logit = torch.softmax(logit, dim=-1)
            output = torch.argmax(logit)
            logit = logit.cpu().detach().numpy()
            none_ground_truth_highest_value = np.max(logit[0, none_ground_truth])
            y_value = logit[0, y]
            mf_score = none_ground_truth_highest_value - y_value
            if mf_score > mf:
                mf = mf_score

            changed_nodes.append(dist.item())
            outputs.append(output.item())

            if not (output == y):
                # print('True Label', y, 'After Attack', output.item(), 'Perturb #', dist.item(), file=self.log_f,
                # flush=True)
                # if dist.item() <= budget:
                succ_rate = 1
                break

        print('avg changed nodes:', sum(changed_nodes) / len(changed_nodes))
        print('success rate:', succ_rate)
        print('time:', time.time() - st)

        return succ_rate, changed_nodes, mf

    def generate(self, inputs, ys, budget, eval_num, alpha=10, epsilon=0.4):
        print()
        st = time.time()
        one_hot_samples = input_process(inputs, self.Dataset)
        self.prob = torch.clone(one_hot_samples) * omegas[self.Dataset]
        self.prob.requires_grad = True
        for p in self.model.parameters():
            p.requires_grad = False
        for k in range(self.itermax):
            # print('pgd step ' + str(k))
            grad = Expect_GumbelSM_grad_batch(self.model, self.prob, one_hot_samples, ys, alpha, epsilon, self.Dataset)

            self.prob = self.prob + self.lr * torch.sign(grad)
            self.prob = torch.clip(self.prob, min=1e-3, max=15)
            self.prob.detach()
            self.prob.requires_grad_
        self.model.eval()
        prob3 = self.prob.repeat(eval_num, 1, 1, 1)
        z = nn.functional.gumbel_softmax(prob3, dim=3, tau=1, hard=False)

        outputs = torch.tensor([]).to(self.prob.device)

        for j in range(eval_num):
            dist = torch.sum(torch.abs(z[j] - one_hot_samples), dim=(1, 2)) / 2
            with torch.no_grad():
                output = self.model(z[j])
            output_y = output[range(len(output)), ys]
            output_y[dist > budget] = 1
            output_y = output_y.unsqueeze(0)
            outputs = torch.cat((outputs, output_y), dim=0)
            print(dist)

        z_idx = torch.argmin(outputs, dim=0)
        z = z[z_idx, range(len(z_idx))]

        # sample adversarial examples from z
        adv_samples = torch.distributions.Categorical(probs=z).sample()
        return adv_samples


def main():
    # creating parser object

    acc_file = np.zeros(5)
    adv_acc_file = np.zeros(5)
    Dataset = args.dataset
    Model_Type = args.modeltype
    budget = budgets[Dataset]
    time_limit = PCAA_time_limits[Dataset]
    t = True
    if args.t == 'False':
        t = False
    print(Dataset, Model_Type)
    output_file = './Logs/%s/%s/' % (Dataset, Model_Type)
    make_dir(output_file)
    X, y = load_data(Dataset, test=t)
    for idx in range(5):
        best_parameters_file = './classifier/{}_{}_{}.par'.format(Dataset, Model_Type, str(idx))

        robust_flag_all = []
        changed_nodes_all = []
        time_all = []

        log_attack = open(
            './Logs/%s/%s/pcaa_Attack_%d.bak' % (Dataset, Model_Type, idx), 'w+')
        attacker = Attacker(best_parameters_file, log_attack, Dataset, Model_Type, itermax=itermaxs[Dataset],
                            lr=lrs[Dataset])
        robust = 0
        success = 0
        mf_all = []
        for i in range(len(X)):
            print(i)
            print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

            sample = X[i]
            label = int(y[i])

            print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

            print("* Original: " + str(sample), file=log_attack, flush=True)

            print("  Original label: %d" % label, file=log_attack, flush=True)

            st = time.time()
            robust_flag = 1
            suc_flag, num_changed, mf_score = attacker.attack(sample, label, budget, eval_num=500)
            mf_score = float(mf_score)
            if suc_flag == -1:
                print('Original Classification Error', file=log_attack, flush=True)
                robust_flag = -1
            else:
                print("* Result: ", file=log_attack, flush=True)
            et = time.time()
            all_t = et - st

            if suc_flag == 0:
                print("This sample is robust.", file=log_attack, flush=True)
                robust += 1
                robust_flag = 1

            if suc_flag == 1:
                print("Successful Number of changed codes: %f" % num_changed[-1], file=log_attack, flush=True)
                robust_flag = 0
                success += 1
            if suc_flag != -1:
                print("Avg Number of changed codes: %f" % np.mean(num_changed), file=log_attack, flush=True)
                print(" Time: " + str(all_t), file=log_attack, flush=True)
                print(" Adv acc: " + str(robust / (i + 1)), file=log_attack, flush=True)

                changed_nodes_all.append(num_changed)
                robust_flag_all.append(robust_flag)
            time_all.append(all_t)
            mf_all.append(mf_score)
            print('mf_score:', mf_score, file=log_attack, flush=True)
            print('mf_avg:', np.mean(mf_all), file=log_attack, flush=True)

        lock = FileLock("./Logs/%s/PCAA_mf.json.lock" % Dataset)
        with lock:
            if os.path.exists('./Logs/%s/PCAA_mf.json' % Dataset):
                mf = json.load(open('./Logs/%s/PCAA_mf.json' % Dataset, 'r'))
            else:
                mf = {}

            if Model_Type in mf.keys():
                pass
            else:
                mf[Model_Type] = {}

            if str(budget) in mf[Model_Type].keys():
                pass
            else:
                mf[Model_Type][str(budget)] = {'avg': {}, 'asr': {}}

            mf[Model_Type][str(budget)]['avg'][str(idx)] = np.mean(mf_all)
            mf[Model_Type][str(budget)]['asr'][str(idx)] = success / (success + robust)
            if list(mf[Model_Type][str(budget)]['avg'].keys()) == [str(i) for i in range(5)]:
                mf[Model_Type][str(budget)]['mean'] = np.mean(
                    [mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
                mf[Model_Type][str(budget)]['std'] = np.std(
                    [mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
                mf[Model_Type][str(budget)]['asr_avg'] = np.mean(
                    [mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
                mf[Model_Type][str(budget)]['asr_std'] = np.std(
                    [mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
            json.dump(mf, open('./Logs/%s/PCAA_mf.json' % Dataset, 'w+'))

        time_attack = []
        robust = 0
        acc = 0
        for j in range(len(robust_flag_all)):
            if robust_flag_all[j] == 0:
                time_attack.append(time_all[j])
                acc += 1
            elif robust_flag_all[j] == 1:
                robust += 1
                acc += 1

        print('average attack_utils time:', np.mean(time_attack), file=log_attack, flush=True)
        print('clean test data accuracy:', len(robust_flag_all) / len(time_all), file=log_attack, flush=True)
        print('adv test data accuracy:', robust / len(time_all), file=log_attack, flush=True)

        acc_file[idx] = acc / len(time_all)
        adv_acc_file[idx] = robust / len(time_all)
    pickle.dump(acc_file, open(output_file + 'acc.pkl', 'wb'))
    pickle.dump(adv_acc_file, open(output_file + 'pcaa_adv_acc.pkl', 'wb'))
    log_all = open('./Logs/%s/PCAA_Attack_all.bak' % Dataset, 'a')
    print('Model:', Model_Type, file=log_all, flush=True)
    print('acc:', acc_file, round(np.mean(acc_file) * 100, 1), '±', round(np.std(acc_file) * 100, 1), file=log_all,
          flush=True)
    print('adv_acc:', adv_acc_file, round(np.mean(adv_acc_file) * 100, 1), '±', round(np.std(adv_acc_file) * 100, 1),
          file=log_all, flush=True)
    print('', file=log_all, flush=True)


if __name__ == '__main__':
    main()
