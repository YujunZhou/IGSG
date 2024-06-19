import time
import argparse

from filelock import FileLock
from advertorch.context import ctx_noparamgrad_and_eval
from utils.utils import *

# creating parser object
parser = argparse.ArgumentParser(description='GradAttack')
parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='census', type=str, help='dataset')
parser.add_argument('--modeltype', default='Transformer_Normal', type=str, help='model type')
parser.add_argument('--eps', default=0.2, type=float, help='PGD epsilon')
parser.add_argument('--t', default='True', type=str, help='test set or whole set')
parser.add_argument('--idx', default=0, type=int, help='running index')
parser.add_argument('--ucbloop', default=20, type=int, help='ucb_loop')
parser.add_argument('--alpha', default=8, type=int, help='alpha')
args = parser.parse_args()

# There are two datasets, some models have the same name for the two dataset, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.PEDecModels import *
elif args.dataset == 'census':
    from Models.CensusModels import *
else:
    raise ValueError('Invalid dataset name')

class Attacker(object):
    def __init__(self, best_parameters_file, log_f):
        # the classes of the dataset
        self.n_categories = num_avail_category[Dataset]
        self.n_labels = num_classes[Dataset]
        self.n_con_fea = num_con_feature[Dataset]
        # load the model
        if 'Transformer' in Model_Type:
            if 'AFD' in Model_Type:
                E = Transformer_E()
                self.model = Transformer_Dc(E)
            else:
                self.model = Transformer()
        else:
            if 'AFD' in Model_Type:
                E = MLP_E()
                self.model = MLP_Dc(E)
            else:
                self.model = MLP()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # load the trained parameters of the classifier
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        self.adversary = LinfPGDAttack_mixed(self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
                                             nb_iter=20, eps_iter=args.eps / 10, rand_init=False, clip_min=-np.inf,
                                             clip_max=np.inf, targeted=False)

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.FloatTensor(np.array([funccall]))
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, Dataset)
        return t_diagnosis_codes

    def classify(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes[0], weight_of_embed_codes[1])
        if 'Transformer' in Model_Type:
            logit = torch.softmax(logit, dim=1)
        logit = logit.cpu()
        # get the prediction
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        g = logit[0][y]
        # find the largest prediction in the false labels
        h = max([logit[0][false_class] for false_class in list_label_set])
        return pred, g, h

    def perturb(self, funccall, arm_chain):
        new_words = copy.deepcopy(funccall)
        for arm in arm_chain:
            new_words[arm[0]] = arm[1]
        return new_words

    def Grad_index(self, funccall, y):
        self.model.train()
        self.model.apply(fix_bn)
        weight_of_embed_codes = self.input_handle(funccall)
        weight_of_embed_codes[1] = weight_of_embed_codes[1].repeat(2, 1, 1)
        weight_of_embed_codes[0] = weight_of_embed_codes[0].repeat(2, 1)
        batch_labels = torch.LongTensor([y]).repeat(2).cuda()
        with ctx_noparamgrad_and_eval(self.model):
            weight_of_embed_codes[0] = self.adversary.perturb(weight_of_embed_codes[0], weight_of_embed_codes[1],
                                                              batch_labels).detach()
        weight_of_embed_codes[1] = torch.autograd.Variable(weight_of_embed_codes[1], requires_grad=True).cuda()
        logit = self.model(weight_of_embed_codes[0], weight_of_embed_codes[1])
        loss = self.criterion(logit, batch_labels)
        loss.backward()
        grad = weight_of_embed_codes[1].grad.cpu().data[0]
        grad = torch.abs(grad)
        grad_norm = torch.norm(grad, dim=1)
        self.model.zero_grad()
        self.model.eval()
        return grad_norm

    def find_candidate(self, new_words, poses, funcall_change):
        poses = poses + self.n_con_fea
        candidates = []
        candidates_change = []
        for candi_sample_index in poses:
            new_candidates = []
            new_candidates_change = []
            for j in range(self.n_categories):
                if j == new_words[candi_sample_index]:
                    continue
                if Dataset in complex_categories.keys():
                    if j >= complex_categories[Dataset][candi_sample_index - self.n_con_fea]:
                        break
                corrupted = copy.deepcopy(new_words)
                corrupted[candi_sample_index] = j
                new_candidates.append(corrupted)
                new_candidates_change.append(1)
            candidates += new_candidates
            candidates_change += new_candidates_change
        candidates.append(new_words)
        candidates_change.append(funcall_change)

        return candidates, candidates_change

    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall != new_funccall)[0])
        return diff_set

    def attack(self, funccall, y):
        print()
        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        current_object = orig_h - orig_g
        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return robust_flag

        n_feature = num_feature[Dataset]
        N_REP = N_REPLACE

        start_random = time.time()
        iteration = 0
        time_Dur = 0
        robust_flag = 1
        greedy_set = set()
        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        funccall_change = 0

        while robust_flag == 1 and len(greedy_set) < budget and time_Dur <= time_limit:
            grad_norm = self.Grad_index(funccall, y)
            for i in greedy_set:
                grad_norm[i] = -1
            topk_feature_index = np.argsort(grad_norm)[-1:]
            candidates, candidates_change = self.find_candidate(funccall, topk_feature_index, funccall_change)
            batch_size = 512
            n_batches = int(np.ceil(float(len(candidates)) / float(batch_size)))
            max_object = -np.inf
            max_index = 0
            for index in range(n_batches):  # n_batches

                batch_diagnosis_codes = torch.FloatTensor(
                    np.array(candidates[batch_size * index: batch_size * (index + 1)]))
                t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
                with ctx_noparamgrad_and_eval(self.model):
                    t_diagnosis_codes[0] = self.adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                                  torch.LongTensor(
                                                                      [y] * t_diagnosis_codes[0].shape[
                                                                          0]).cuda()).detach()
                logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
                if 'Transformer' in Model_Type:
                    logit = torch.softmax(logit, dim=1)
                logit = logit.data.cpu().numpy()
                subsets_g = logit[:, y]
                subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
                subsets_object = subsets_h - subsets_g
                # get the maximum object, and update worst object
                temp_max_object = np.max(subsets_object)
                temp_max_index = np.argmax(subsets_object) + batch_size * index

                if temp_max_object > max_object:
                    max_object = temp_max_object
                    max_index = temp_max_index
            if max_object > 0:
                robust_flag = 0
                break
            changed_set = self.changed_set(funccall[self.n_con_fea:], candidates[max_index][self.n_con_fea:])
            funccall_change = len(changed_set) + funccall_change
            funccall = candidates[max_index]
            for i in changed_set:
                if i not in greedy_set:
                    greedy_set.add(i)
            iteration += 1
            time_Dur = time.time() - start_random
        return robust_flag

Dataset = args.dataset
if Dataset == 'pedec':
    N_REPLACE = 100
else:
    N_REPLACE = 10
Model_Type = args.modeltype
if args.budget == 0:
    budget = budgets[Dataset]
else:
    budget = args.budget
time_limit = OMPGS_time_limits[Dataset]*10
t = True
if args.t == 'False':
    t = False

print(Dataset, Model_Type, t)
output_file = './Logs/%s/%s/' % (Dataset, Model_Type)
if os.path.isdir(output_file):
    pass
else:
    os.mkdir(output_file)
if not t:
    output_file += 'all_'

X, y = load_data(Dataset, test=t)
scaler = pickle.load(open('./dataset/' + Dataset + '_scaler.pkl', 'rb'))
X[:, :num_con_feature[Dataset]] = scaler.transform(X[:, :num_con_feature[Dataset]])
for args.idx in range(5):
    best_parameters_file = './classifier/{}_{}_{}.par'.format(Dataset, Model_Type, str(args.idx))

    log_attack = open(
        './Logs/%s/%s/GradAttack_bgt_%d_%d.bak' % (Dataset, Model_Type, budget, args.idx), 'w+')

    attacker = Attacker(best_parameters_file, log_attack)

    robust = 0
    success = 0
    for i in range(len(y)):
        print(i)
        print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

        sample = X[i]
        label = int(y[i])

        print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

        print("* Original: " + str(sample), file=log_attack, flush=True)

        print("  Original label: %d" % label, file=log_attack, flush=True)

        st = time.time()
        robust_flag = attacker.attack(sample, label)
        # print("Orig_Prob = " + str(g_process[0]), file=log_attack, flush=True)
        if robust_flag == -1:
            print('Original Classification Error', file=log_attack, flush=True)
        else:
            print("* Result: ", file=log_attack, flush=True)
        et = time.time()
        all_t = et - st

        if robust_flag == 1:
            print("This sample is robust.", file=log_attack, flush=True)
            robust += 1
        elif robust_flag == 0:
            success += 1

        if robust_flag != -1:
            print(" Time: " + str(all_t), file=log_attack, flush=True)
            print(" Adv acc: " + str(robust/(i+1)), file=log_attack, flush=True)

    lock = FileLock("./Logs/%s/GradAttack_mf.json.lock" % Dataset)
    with lock:
        if os.path.exists('./Logs/%s/GradAttack_mf.json' % Dataset):
            mf = json.load(open('./Logs/%s/GradAttack_mf.json' % Dataset, 'r'))
        else:
            mf = {}

        if Model_Type in mf.keys():
            pass
        else:
            mf[Model_Type] = {}

        if str(budget) in mf[Model_Type].keys():
            pass
        else:
            mf[Model_Type][str(budget)] = {'advacc': {}, 'asr': {}}

        mf[Model_Type][str(budget)]['asr'][str(args.idx)] = success / (success + robust)
        mf[Model_Type][str(budget)]['advacc'][str(args.idx)] = robust / len(y)
        if list(mf[Model_Type][str(budget)]['asr'].keys()) == [str(i) for i in range(5)]:
            mf[Model_Type][str(budget)]['asr_avg'] = np.mean([mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['asr_std'] = np.std([mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['advacc_avg'] = np.mean([mf[Model_Type][str(budget)]['advacc'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['advacc_std'] = np.std([mf[Model_Type][str(budget)]['advacc'][str(i)] for i in range(5)])

        json.dump(mf, open('./Logs/%s/GradAttack_mf.json' % Dataset, 'w+'))
