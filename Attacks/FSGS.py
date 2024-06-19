import time
from itertools import combinations
import argparse

from filelock import FileLock

from utils.utils import *

# creating parser object
parser = argparse.ArgumentParser(description='FSGS')
parser.add_argument('--budget', default=1, type=int, help='purturb budget')
parser.add_argument('--dataset', default='Splice', type=str, help='dataset')
parser.add_argument('--modeltype', default='MLP_IGSG', type=str, help='model type')
# parser.add_argument('--time', default=30, type=int, help='time limit')
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
    print('Dataset not supported')
    exit()


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

    def funccall_query(self, eval_funccall, greedy_set):
        candidate_lists = []
        funccall_lists = []

        for i in range(min(len(greedy_set) + 1, budget)):
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

        return funccall_lists

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

        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
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
            success_flag = 1
            pos_dict = {}
            funccall_lists_all = []

            # we loop over each feature and each category to find the worst object and its position
            for visit_idx in range(len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                for code_idx in range(num_avail_category[Dataset]):
                    if code_idx == funccall[visit_idx]:
                        continue
                    if Dataset in complex_categories.keys():
                        if code_idx >= complex_categories[Dataset][visit_idx]:
                            break
                    pos = (visit_idx, code_idx)
                    eval_funccall = copy.deepcopy(funccall)
                    eval_funccall[visit_idx] = code_idx
                    funccall_list_temp = self.funccall_query(eval_funccall, greedy_set)

                    funccall_lists_all = funccall_lists_all + funccall_list_temp
                    pos_dict[len(funccall_lists_all)] = pos

            query_num += len(funccall_lists_all)
            batch_size = 512
            n_batches = int(np.ceil(float(len(funccall_lists_all)) / float(batch_size)))
            max_object = -np.inf
            max_index = 0
            for index in range(n_batches):  # n_batches

                batch_diagnosis_codes = torch.LongTensor(np.array(funccall_lists_all[batch_size * index: batch_size * (index + 1)]))
                t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
                logit = self.model(t_diagnosis_codes)
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
            poses = np.array(list(pos_dict.keys()))
            max_pos_index = np.where(poses > max_index)[0][0]
            max_pos = pos_dict[poses[max_pos_index]]
            greedy_set_best_temp_funccall = funccall_lists_all[max_index]

            print(iteration)
            print('query', query_num)
            print(max_object)

            greedy_set.add(max_pos)
            greedy_set_visit_idx.add(max_pos[0])
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))
            if max_object > current_object:
                current_object = max_object
            if max_object > 0:
                success_flag = 0
            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))

            print(greedy_set)
            if success_flag == 1:
                # if (time.time() - st) > time_limit or len(greedy_set) == num_feature[Dataset]:
                if iteration == budget or len(greedy_set) == num_feature[Dataset]:
                    success_flag = -1
                    robust_flag = 1
                    print('Time out')

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)
            print('Attack successfully')

        print(time.time()-st)
        print("Modified_set:", flip_set)
        print(flip_funccall)

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
time_limit = FSGS_time_limits[Dataset]
t = True
if args.t == 'False':
    t = False

print(Dataset, Model_Type)
output_file = './Logs/%s/%s/' % (Dataset, Model_Type)
make_dir(output_file)

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
        './Logs/%s/%s/greedmax_Attack_bgt_%d_%d.bak' % (Dataset, Model_Type, budget, args.idx), 'w+')


    attacker = Attacker(best_parameters_file, log_attack)
    robust = 0
    for i in range(len(X)):
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

    lock = FileLock("./Logs/%s/FSGS_mf.json.lock" % Dataset)
    with lock:
        if os.path.exists('./Logs/%s/FSGS_mf.json' % Dataset):
            mf = json.load(open('./Logs/%s/FSGS_mf.json' % Dataset, 'r'))
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
            mf[Model_Type][str(budget)]['mean'] = np.mean(
                [mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['std'] = np.std([mf[Model_Type][str(budget)]['avg'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['freq'] = list(feature_dict.values())
            mf[Model_Type][str(budget)]['asr_avg'] = np.mean(
                [mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
            mf[Model_Type][str(budget)]['asr_std'] = np.std(
                [mf[Model_Type][str(budget)]['asr'][str(i)] for i in range(5)])
        json.dump(mf, open('./Logs/%s/FSGS_mf.json' % Dataset, 'w+'))

    acc = len(robust_flag_all) / len(X)
    adv_acc = robust / len(X)
    # acc_file[args.idx] = acc
    if os.path.exists('./Logs/%s/%s/fsgs_adv_acc.pkl' % (Dataset, Model_Type)):
        adv_acc_file = pickle.load(open('./Logs/%s/%s/fsgs_adv_acc.pkl' % (Dataset, Model_Type), 'rb'))
    else:
        adv_acc_file = np.zeros(5)
    adv_acc_file[args.idx] = adv_acc
    pickle.dump(adv_acc_file, open(output_file + 'fsgs_adv_acc.pkl', 'wb'))
json.dump(feature_dict, open('./Logs/%s/%s/FSGS_feature_dict_%d.json' % (Dataset, Model_Type, budget), 'w+'))


