import time
from itertools import combinations
import argparse
from filelock import FileLock

from utils.utils import *

# creating parser object
parser = argparse.ArgumentParser(description='OMPGS')
parser.add_argument('--budget', default=3, type=int, help='purturb budget')
parser.add_argument('--dataset', default='Splice', type=str, help='dataset')
parser.add_argument('--modeltype', default='MLP_Normal', type=str, help='model type')
# parser.add_argument('--time', default=3, type=int, help='time limit')
parser.add_argument('--t', default='True', type=str, help='test set or whole set')
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
    raise ValueError('Dataset not found')


class Attacker(object):
    def __init__(self, best_parameters_file, log_f):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        # load the model
        if 'Transformer' in Model_Type:
            if 'AFD' in Model_Type:
                E, Dc = Transformer_E(), Transformer_Dc()
                self.model = nn.Sequential(E, Dc)
            else:
                self.model = Transformer()
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
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.LongTensor(np.array([funccall]))
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, Dataset)
        return t_diagnosis_codes

    def classify(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes)
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

    def eval_object(self, eval_funccall, greedy_set, orig_label, greedy_set_visit_idx, query_num,
                    greedy_set_best_temp_funccall):
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(orig_label)
        list_label_set = list(label_set)
        flip_set = set()
        flip_funccall = torch.tensor([])

        # candidate_lists contains all the non-empty subsets of greedy_set
        for i in range(0, min(len(greedy_set) + 1, budget)):
            subset1 = combinations(greedy_set, i)
            for subset in subset1:
                candidate_lists.append(list(subset))

        # change the funccall based on the candidates above and get the candidate funccalls.
        for can in candidate_lists:
            temp_funccall = copy.deepcopy(eval_funccall)
            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)
        query_num += len(funccall_lists)
        batch_size = 64
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
        max_subsets_object = -np.inf
        max_subset_index = -1
        grad_feature_list = torch.tensor([])
        grad_cate_index_list = torch.tensor([], dtype=torch.long)
        # first, we eval all the candidates and get the gradients, and then we find the largest gradient candidate
        # and category for each feature
        for index in range(n_batches):  # n_batches
            self.model.eval()
            batch_diagnosis_codes = torch.LongTensor(np.array(funccall_lists[batch_size * index: batch_size * (index + 1)]))
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            logit = self.model(t_diagnosis_codes)
            if 'Transformer' in Model_Type:
                logit = torch.softmax(logit, dim=1)
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
            subsets_objects = subsets_h - subsets_g
            max_subset_object_temp = max(subsets_objects)
            if max_subset_object_temp > max_subsets_object:
                max_subsets_object = max_subset_object_temp
                max_subset_index = batch_size * index + np.argmax(subsets_objects)

            self.model.train()
            self.model.apply(fix_bn)
            grad_all = torch.tensor([])
            flag = 0
            for i in range(len(list_label_set)):
                flag = 0
                self.model.zero_grad()
                t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
                batch_labels = torch.tensor([list_label_set[i]] * len(batch_diagnosis_codes)).cuda()
                if t_diagnosis_codes.size(0) == 1:
                    flag = 1
                    if Dataset_type[Dataset] == 'multi':
                        t_diagnosis_codes = t_diagnosis_codes.repeat(2, 1, 1)
                    else:
                        t_diagnosis_codes = t_diagnosis_codes.repeat(2, 1)
                    batch_labels = batch_labels.repeat(2)
                t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
                logit = self.model(t_diagnosis_codes)
                loss = self.criterion(logit, batch_labels)
                loss.backward()
                # we use the gradient of the false label. since there are only 3 lables, we just use grad_0 and _1
                grad = t_diagnosis_codes.grad.cpu().data
                # for Splice, there is a invalid category, and we need to remove it.
                grad = torch.abs(grad)
                # print(grad_0[:, 0].norm(dim=0))
                grad_all = torch.cat((grad_all, grad.unsqueeze(0)), dim=0)

            self.model.zero_grad()
            grad = torch.max(grad_all, dim=0)[0]
            if flag == 1:
                grad = grad[0].unsqueeze(0)
            subsets_g = subsets_g.reshape(-1, 1)
            subsets_g = torch.tensor(subsets_g)
            if Dataset_type[Dataset] == 'multi':
                grad_feature_temp = torch.max(grad, dim=2)[0]
                grad_feature_temp = grad_feature_temp / subsets_g
                grad_cate_index = torch.argmax(grad, dim=2)
                grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=0)
            else:
                grad_feature_temp = grad / subsets_g
            grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=0)

        # if the one of the candidates attacks successfully, then we exit.
        if max_subsets_object >= 0 or len(greedy_set) == num_feature[Dataset]:
            if max_subsets_object >= 0:
                print(max_subsets_object)
                success_flag = 0
                flip_funccall = copy.deepcopy(funccall_lists[max_subset_index])
                greedy_set_best_temp_funccall = copy.deepcopy(funccall_lists[max_subset_index])
                flip_set = self.changed_set(eval_funccall, flip_funccall)
            else:
                # success flag = -2 means we have attacked all the features.
                success_flag = -2
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                   greedy_set_visit_idx, flip_set, flip_funccall, query_num

        self.model.eval()
        grad_feature, grad_set_index_list = torch.max(grad_feature_list, dim=0)
        top_100_features = torch.argsort(grad_feature, descending=True)[:100]
        funccalls = []
        features = []
        # for each feature, we choose the optimal candidate and optimal category and then we run the exactly
        # and pick the largest.
        for index in top_100_features:
            if index.item() in greedy_set_visit_idx:
                continue
            temp_funccall = copy.deepcopy(funccall_lists[grad_set_index_list[index]])
            if Dataset_type[Dataset] == 'multi':
                temp_funccall[index] = grad_cate_index_list[grad_set_index_list[index], index]
                if Dataset in complex_categories.keys():
                    if temp_funccall[index] >= complex_categories[Dataset][index]:
                        print('!!!', index, temp_funccall[index], '!!!')
                        continue
            elif Dataset_type[Dataset] == 'binary':
                temp_funccall[index] = 1 - temp_funccall[index]
            else:
                pass
            features.append(index)
            funccalls.append(temp_funccall)

        if not funccalls:
            success_flag = -2
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                   greedy_set_visit_idx, flip_set, flip_funccall, query_num

        funccalls = torch.LongTensor(np.array(funccalls))
        query_num += len(features)
        t_diagnosis_codes = input_process(funccalls, Dataset)
        logit = self.model(t_diagnosis_codes)
        if 'Transformer' in Model_Type:
            logit = torch.softmax(logit, dim=1)
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
        objects = h - g

        max_object = np.max(objects)
        max_index = np.argmax(objects)

        max_feature = features[max_index].item()
        if Dataset_type[Dataset] == 'multi':
            max_category = grad_cate_index_list[grad_set_index_list[max_feature], max_feature].item()
        elif Dataset_type[Dataset] == 'binary':
            max_category = int(1 - eval_funccall[max_feature])
        else:
            max_category = None
        # if the max object changs, we update it and the best funccall
        if max_object < max_subsets_object:
            max_object = max_subsets_object
            greedy_set_best_temp_funccall = funccall_lists[max_subset_index]
        else:
            max_set = grad_set_index_list[max_feature]
            greedy_set_best_temp_funccall = copy.deepcopy(funccall_lists[max_set])
            greedy_set_best_temp_funccall[max_feature] = max_category

        if max_object >= 0:
            success_flag = 0
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(eval_funccall, flip_funccall)

        # update the greedy set
        greedy_set_visit_idx.add(max_feature)
        greedy_set.add((max_feature, max_category))

        return max_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
               flip_set, flip_funccall, query_num

    # calculate which feature is changed
    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall != new_funccall)[0])
        return diff_set

    def attack(self, funccall, y):
        print()
        st = time.time()
        success_flag = 1

        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0
        query_num = 0

        current_object = orig_h - orig_g
        flip_funccall = funccall

        # when the classification is wrong for the original example, skip the attack_utils
        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                   query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                   greedy_set_best_temp_funccall, \
                   n_changed, flip_funccall, flip_set, iteration

        print(current_object)
        # once the success_flag ==0, we attack_utils successfully and exit
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
            flip_set, flip_funccall, query_num = self.eval_object(funccall, greedy_set, y,
                                                                  greedy_set_visit_idx, query_num,
                                                                  greedy_set_best_temp_funccall)

            print(iteration)
            print(worst_object)
            print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(worst_object)
            greedy_set_process.append(copy.deepcopy(greedy_set))

            # time limit exceed or we have attacked all the features, but it is still not successful.
            # if (time.time() - st) > time_limit or success_flag == -2:
            if iteration == budget or success_flag == -2:
                success_flag = -1
                robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
               query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
               greedy_set_best_temp_funccall, \
               n_changed, flip_funccall, flip_set, iteration


Dataset = args.dataset
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
feature_dict = {}
for i in range(num_feature[Dataset]):
    feature_dict[i] = 0
for args.idx in range(5):
    best_parameters_file = './classifier/{}_{}_{}.par'.format(Dataset, Model_Type, str(args.idx))

    # ready for attack_utils and log the files
    g_process_all = []
    mf_process_all = []
    greedy_set_process_all = []
    changed_set_process_all = []

    query_num_all = []
    robust_flag_all = []

    orignal_funccalls_all = []
    orignal_labels_all = []

    final_greedy_set_all = []
    final_greedy_set_visit_idx_all = []
    final_changed_num_all = []
    final_funccall_all = []

    flip_funccall_all = []
    flip_set_all = []
    flip_mf_all = []
    flip_sample_original_label_all = []
    flip_sample_index_all = []
    mf_all = []

    iteration_all = []
    time_all = []

    log_attack = open(
        './Logs/%s/%s/gradmax_Attack_bgt_%d_%d.bak' % (Dataset, Model_Type, budget, args.idx), 'w+')

    attacker = Attacker(best_parameters_file, log_attack)

    robust = 0
    for i in range(len(y)):
        print(i)
        print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

        sample = X[i]
        label = int(y[i])

        print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

        print("* Original: " + str(sample), file=log_attack, flush=True)

        print("  Original label: %d" % label, file=log_attack, flush=True)

        st = time.time()
        g_process, mf_process, greedy_set_process, changed_set_process, query_num, robust_flag, \
        greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, \
        num_changed, flip_funccall, flip_set, iteration = attacker.attack(sample, label)
        print("Orig_Prob = " + str(g_process[0]), file=log_attack, flush=True)
        if robust_flag == -1:
            print('Original Classification Error', file=log_attack, flush=True)
        else:
            print("* Result: ", file=log_attack, flush=True)
        et = time.time()
        all_t = et - st

        if robust_flag == 1:
            print("This sample is robust.", file=log_attack, flush=True)
            robust += 1

        if robust_flag != -1:
            print('g_process:', g_process, file=log_attack, flush=True)
            print('mf_process:', mf_process, file=log_attack, flush=True)
            print('greedy_set_process:', greedy_set_process, file=log_attack, flush=True)
            print('changed_set_process:', changed_set_process, file=log_attack, flush=True)
            print("  Number of query for this: " + str(query_num), file=log_attack, flush=True)
            print('greedy_set: ', file=log_attack, flush=True)
            print(greedy_set, file=log_attack, flush=True)
            print('greedy_set_visit_idx: ', file=log_attack, flush=True)
            print(greedy_set_visit_idx, file=log_attack, flush=True)
            print('greedy_funccall:', file=log_attack, flush=True)
            print(greedy_set_best_temp_funccall, file=log_attack, flush=True)
            print('best_prob = ' + str(g_process[-1]), file=log_attack, flush=True)
            print('best_object = ' + str(mf_process[-1]), file=log_attack, flush=True)
            print("  Number of changed codes: %d" % num_changed, file=log_attack, flush=True)
            print("risk funccall:", file=log_attack, flush=True)
            print('iteration: ' + str(iteration), file=log_attack, flush=True)
            print(" Time: " + str(all_t), file=log_attack, flush=True)
            if robust_flag == 0:
                print('flip_funccall:', file=log_attack, flush=True)
                print(flip_funccall, file=log_attack, flush=True)
                print('flip_set:', file=log_attack, flush=True)
                print(flip_set, file=log_attack, flush=True)
                print('flip_object = ', mf_process[-1], file=log_attack, flush=True)
                print(" The cardinality of S: " + str(len(greedy_set)), file=log_attack, flush=True)
            else:
                print(" The cardinality of S: " + str(len(greedy_set)) + ', but timeout', file=log_attack,
                      flush=True)
            print(" Adv acc: " + str(robust/(i+1)), file=log_attack, flush=True)

            time_all.append(all_t)
            g_process_all.append(copy.deepcopy(g_process))
            mf_process_all.append(copy.deepcopy(mf_process))
            greedy_set_process_all.append(copy.deepcopy(greedy_set_process))
            changed_set_process_all.append(copy.deepcopy(changed_set_process))

            query_num_all.append(query_num)
            robust_flag_all.append(robust_flag)
            iteration_all.append(iteration)

            orignal_funccalls_all.append(copy.deepcopy(X[i].tolist()))
            orignal_labels_all.append(label)

            final_greedy_set_all.append(copy.deepcopy(greedy_set))
            final_greedy_set_visit_idx_all.append(copy.deepcopy(greedy_set_visit_idx))
            final_funccall_all.append(copy.deepcopy(greedy_set_best_temp_funccall))
            final_changed_num_all.append(num_changed)

            if robust_flag == 0:
                flip_funccall_all.append(copy.deepcopy(flip_funccall))
                flip_set_all.append(copy.deepcopy(flip_set))
                flip_mf_all.append(mf_process[-1])
                flip_sample_original_label_all.append(label)
                flip_sample_index_all.append(i)
                for flip_fea in flip_set:
                    feature_dict[flip_fea] += 1

        else:
            final_funccall_all.append(copy.deepcopy(sample))

        mf_score = mf_process[-1]
        mf_all.append(mf_score)
        print('mf_score:', mf_score, file=log_attack, flush=True)
        print('mf_avg:', np.mean(mf_all), file=log_attack, flush=True)
        print('mf_std:', np.std(mf_all), file=log_attack, flush=True)

    lock = FileLock("./Logs/%s/OMPGS_mf.json.lock" % Dataset)
    with lock:
        if os.path.exists('./Logs/%s/OMPGS_mf.json' % Dataset):
            mf = json.load(open('./Logs/%s/OMPGS_mf.json' % Dataset, 'r'))
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

        mf[Model_Type][str(budget)]['avg'][str(args.idx)] = np.mean(mf_all)
        mf[Model_Type][str(budget)]['asr'][str(args.idx)] = len(flip_set_all) / len(time_all)
        if list(mf[Model_Type][str(budget)]['avg'].keys()) == [str(i) for i in range(5)]:
            mf[Model_Type][str(budget)]['mean'] = np.mean([mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['std'] = np.std([mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['freq'] = list(feature_dict.values())
            mf[Model_Type][str(budget)]['asr_avg'] = np.mean([mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['asr_std'] = np.std([mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])

        json.dump(mf, open('./Logs/%s/OMPGS_mf.json' % Dataset, 'w+'))

    acc = len(robust_flag_all) / len(X)
    adv_acc = robust / len(X)
    if os.path.exists('./Logs/%s/%s/acc.pkl' % (Dataset, Model_Type)):
        acc_file = pickle.load(open('./Logs/%s/%s/acc.pkl' % (Dataset, Model_Type), 'rb'))
    else:
        acc_file = np.zeros(5)
    if os.path.exists('./Logs/%s/%s/ompgs_adv_acc.pkl' % (Dataset, Model_Type)):
        adv_acc_file = pickle.load(open('./Logs/%s/%s/ompgs_adv_acc.pkl' % (Dataset, Model_Type), 'rb'))
    else:
        adv_acc_file = np.zeros(5)
    acc_file[args.idx] = acc
    adv_acc_file[args.idx] = adv_acc
    pickle.dump(acc_file, open(output_file + 'acc.pkl', 'wb'))
    pickle.dump(adv_acc_file, open(output_file + 'ompgs_adv_acc.pkl', 'wb'))
json.dump(feature_dict, open('./Logs/%s/%s/OMPGS_feature_dict_%d.json' % (Dataset, Model_Type, budget), 'w+'))
