import time
import argparse
from advertorch.context import ctx_noparamgrad_and_eval
from filelock import FileLock

from utils.utils import *

# creating parser object
parser = argparse.ArgumentParser(description='FEAT MIXED')
parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='census', type=str, help='dataset')
parser.add_argument('--eps', default=0.2, type=float, help='PGD epsilon')
parser.add_argument('--modeltype', default='Transformer_Normal', type=str, help='model type')
# parser.add_argument('--time', default=3, type=int, help='time limit')
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
    print('Dataset not supported')
    exit()


class Attacker(object):
    def __init__(self, best_parameters_file, log_f):
        # the classes of the dataset
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
        grad[range(len(funccall)-self.n_con_fea), funccall[self.n_con_fea:]] = 0
        grad_feature, grad_cate_index = torch.max(grad, dim=1)
        self.model.zero_grad()
        self.model.eval()
        return grad_feature, grad_cate_index, weight_of_embed_codes[0][0]

    def word_paraphrase_nocombine(self, new_words, orig_prob_new, poses, category_set, y):
        index_candidate = new_words[poses+self.n_con_fea]
        pred_candidate = np.array(len(poses) * [orig_prob_new])

        candidates = []
        for i, candi_sample_index in enumerate(poses):
            corrupted = copy.deepcopy(new_words)
            corrupted[candi_sample_index+self.n_con_fea] = category_set[candi_sample_index]
            candidates.append(corrupted)

        batch_diagnosis_codes = torch.FloatTensor(np.array(candidates))
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        pred_probs = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
        result = 1 - pred_probs[:, y]
        pred_prob = result.cpu().detach().numpy()

        pred_candidate = np.where(pred_prob >= pred_candidate, pred_prob, pred_candidate)
        index_candidate = np.where(pred_prob >= pred_candidate, category_set[poses], index_candidate)

        return index_candidate, pred_candidate

    def UCBV(self, round, pred_set_list, N):
        mean = pred_set_list / N
        variation = (pred_set_list - mean) ** 2 / N
        delta = np.sqrt((args.alpha * variation * math.log(round)) / N) + (args.alpha * math.log(round) / N)
        ucb = mean + delta

        return ucb

    def attack(self, funccall, y):
        print()

        orig_pred, orig_g, orig_h = self.classify(funccall, y)
        current_object = orig_h - orig_g

        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return robust_flag

        n_feature = num_feature[Dataset]
        RN = 5
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        arm_preds = [1-orig_g]
        N_REP = N_REPLACE

        all_robust_flag = 1

        for n in range(RN):
            start_random = time.time()
            iteration = 0
            time_Dur = 0
            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            visited = []
            robust_flag = 1

            while robust_flag == 1 and len(arm_chain) <= budget and time_Dur <= time_limit:

                new_words = self.perturb(funccall, arm_chain)
                orig_pred_new, orig_prob_new, _ = self.classify(new_words, y)
                orig_prob_new = 1 - orig_prob_new

                grad_set, category_set, funccall_0 = self.Grad_index(new_words, y)
                new_words[:self.n_con_fea] = funccall_0.cpu().data.numpy()
                self.model.eval()
                if torch.sum(grad_set) != 0:
                    WeightProb = np.array(grad_set / torch.sum(grad_set))
                else:
                    WeightProb = np.array([1/n_feature] * n_feature)
                K_set = np.random.choice(range(n_feature), size=N_REP, replace=False, p=WeightProb)

                N = np.ones(len(K_set))
                index_candidate, pred_set_list = self.word_paraphrase_nocombine(new_words, orig_prob_new, K_set,
                                                                                category_set, y)
                for arm in visited:
                    pred_set_list[arm] = -1
                ucb_loop = 0

                while robust_flag == 1 and ucb_loop <= 20 and time_Dur <= time_limit:
                    tmp_arm_chain = []
                    tmp_visited = []
                    ucb_loop = ucb_loop + 1
                    iteration += 1
                    for _ in range(budget - len(arm_chain)):
                        tmp_pred_set_list = copy.deepcopy(pred_set_list)
                        for arm in tmp_visited:
                            tmp_pred_set_list[arm] = -1
                        ucb = self.UCBV(iteration, tmp_pred_set_list, N)
                        topk_feature_index = np.argsort(ucb)[-1]
                        tmp_visited.append(topk_feature_index)
                        tmp_words = self.perturb(new_words, tmp_arm_chain)
                        tmp_pred, tmp_prob, _ = self.classify(tmp_words, y)
                        tmp_prob = 1 - tmp_prob
                        Feat_max = K_set[topk_feature_index]
                        cand_max, pred_max = self.word_paraphrase_nocombine(tmp_words, tmp_prob, np.array([Feat_max]),
                                                                            category_set, y)
                        tmp_arm_chain.append([Feat_max+self.n_con_fea, cand_max[0]])
                        n_add = np.eye(len(N))[topk_feature_index]
                        N += n_add

                        pred_set_list_add = np.zeros(len(K_set))
                        pred_set_list_add[topk_feature_index] = pred_max
                        pred_set_list = pred_set_list + pred_set_list_add

                        time_end = time.time()
                        time_Dur = time_end - start_random
                        if pred_max > TAU:
                            arm_chain = arm_chain + tmp_arm_chain
                            arm_pred = arm_pred + [pred_max]
                            success.append(1)
                            num_armchain.append(len(arm_chain))
                            n_change.append(len(arm_chain))
                            time_success.append(time_Dur)
                            arm_chains.append(arm_chain)
                            arm_preds.append(arm_pred)
                            robust_flag = 0
                            break
                    if pred_max > TAU:
                        break
                    ucb = self.UCBV(iteration, pred_set_list, N)
                    topk_feature_index = np.argsort(ucb)[-1]
                    Feat_max = K_set[topk_feature_index]
                    cand_max, pred_max = self.word_paraphrase_nocombine(new_words, orig_prob_new, np.array([Feat_max]),
                                                                        category_set, y)
                    arm_chain.append([Feat_max+self.n_con_fea, cand_max[0]])
                    arm_pred.append(pred_max)
                    visited.append(topk_feature_index)
            if robust_flag == 0:
                all_robust_flag = 0
                break
        return all_robust_flag

TAU = 0.5
SubK_ratio = 0.3

Dataset = args.dataset
if Dataset == 'pedec':
    N_REPLACE = 100
else:
    N_REPLACE = 20
Model_Type = args.modeltype
if args.budget == 0:
    budget = budgets[Dataset]
else:
    budget = args.budget
time_limit = OMPGS_time_limits[Dataset]
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
# feature_dict = {}
# for i in range(num_feature[Dataset]):
#     feature_dict[i] = 0
for args.idx in range(5):
    best_parameters_file = './classifier/{}_{}_{}.par'.format(Dataset, Model_Type, str(args.idx))

    log_attack = open(
        './Logs/%s/%s/FEAT_Attack_bgt_%d_%d.bak' % (Dataset, Model_Type, budget, args.idx), 'w+')

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

    lock = FileLock("./Logs/%s/FEAT_mf.json.lock" % Dataset)
    with lock:
        if os.path.exists('./Logs/%s/FEAT_mf.json' % Dataset):
            mf = json.load(open('./Logs/%s/FEAT_mf.json' % Dataset, 'r'))
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

        json.dump(mf, open('./Logs/%s/FEAT_mf.json' % Dataset, 'w+'))
