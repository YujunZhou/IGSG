import time
from itertools import combinations

import numpy as np
import torch
from advertorch.context import ctx_noparamgrad_and_eval
from utils import *
import torch.nn.functional as F


class Attacker(object):
    def __init__(self, model, log_f, Dataset):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        self.model = model
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # the number of the categories of the dataset
        self.n_diagonosis_codes = num_category[Dataset]
        self.Dataset = Dataset

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.LongTensor([funccall])
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, self.Dataset)
        return t_diagnosis_codes

    def classify(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes)
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
        for i in range(0, min(len(greedy_set) + 1, budgets[self.Dataset])):
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
            batch_diagnosis_codes = torch.LongTensor(funccall_lists[batch_size * index: batch_size * (index + 1)])
            t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
            logit = self.model(t_diagnosis_codes)
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
                batch_labels = torch.tensor([list_label_set[i]] * len(batch_diagnosis_codes)).cuda()
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                if t_diagnosis_codes.size(0) == 1:
                    flag = 1
                    if Dataset_type[self.Dataset] == 'multi':
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
            if Dataset_type[self.Dataset] == 'multi':
                grad_feature_temp = torch.max(grad, dim=2)[0]
                grad_feature_temp = grad_feature_temp / subsets_g
                grad_cate_index = torch.argmax(grad, dim=2)
                grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=0)
            else:
                grad_feature_temp = grad / subsets_g
            grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=0)

        # if the one of the candidates attacks successfully, then we exit.
        if max_subsets_object >= 0 or len(greedy_set) == num_feature[self.Dataset]:
            if max_subsets_object >= 0:
                # print(max_subsets_object)
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
            if Dataset_type[self.Dataset] == 'multi':
                temp_funccall[index] = int(grad_cate_index_list[grad_set_index_list[index], index].item())
                if self.Dataset in complex_categories.keys():
                    if temp_funccall[index] >= complex_categories[self.Dataset][index]:
                        print('!!!', index, temp_funccall[index], '!!!')
                        continue
            elif Dataset_type[self.Dataset] == 'binary':
                temp_funccall[index] = 1 - temp_funccall[index]
            else:
                pass
            features.append(index)
            funccalls.append(temp_funccall)

        funccalls = torch.LongTensor(funccalls)
        query_num += len(features)
        t_diagnosis_codes = input_process(funccalls, self.Dataset)
        # t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        logit = self.model(t_diagnosis_codes)
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
        objects = h - g

        max_object = np.max(objects)
        max_index = np.argmax(objects)

        max_feature = features[max_index].item()
        if Dataset_type[self.Dataset] == 'multi':
            max_category = grad_cate_index_list[grad_set_index_list[max_feature], max_feature].item()
        elif Dataset_type[self.Dataset] == 'binary':
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
        # print()
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
        time_limit = OMPGS_time_limits[self.Dataset]

        # when the classification is wrong for the original example, skip the attack_utils
        if current_object > 0:
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
        # once the success_flag ==0, we attack_utils successfully and exit
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
                flip_set, flip_funccall, query_num = self.eval_object(funccall, greedy_set, y,
                                                                      greedy_set_visit_idx, query_num,
                                                                      greedy_set_best_temp_funccall)

            # print(iteration)
            # print(worst_object)
            # print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(worst_object)
            greedy_set_process.append(copy.deepcopy(greedy_set))

            # time limit exceed or we have attacked all the features, but it is still not successful.
            # if (time.time() - st) > time_limit or success_flag == -2:
            if iteration == budgets[self.Dataset] or success_flag == -2:
                success_flag = -1
                robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration

    def funccall_query(self, eval_funccall, greedy_set):
        candidate_lists = []
        funccall_lists = []

        for i in range(min(len(greedy_set) + 1, budgets[self.Dataset])):
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

    def attack_FSGS(self, funccall, y):
        st = time.time()
        time_limit = 1
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
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
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
                for code_idx in range(num_avail_category[self.Dataset]):
                    if code_idx == funccall[visit_idx]:
                        continue
                    if self.Dataset in complex_categories.keys():
                        if code_idx >= complex_categories[self.Dataset][visit_idx]:
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

                batch_diagnosis_codes = torch.LongTensor(
                    funccall_lists_all[batch_size * index: batch_size * (index + 1)])
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
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

            # print(iteration)
            # print('query', query_num)
            # print(max_object)

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

            # print(greedy_set)
            if success_flag == 1:
                # if (time.time() - st) > time_limit or len(greedy_set) == num_feature[self.Dataset]:
                if iteration == budgets[self.Dataset] or len(greedy_set) == num_feature[self.Dataset]:
                    success_flag = -1
                    robust_flag = 1
                    # print('Time out')

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)
            # print('Attack successfully')
        #
        # print("Modified_set:", flip_set)
        # print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration


class Attacker_mixed(object):
    def __init__(self, model, log_f, Dataset):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        self.model = model
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # the number of the categories of the dataset
        self.n_diagonosis_codes = num_category[Dataset]
        self.Dataset = Dataset
        self.n_con_fea = num_con_feature[Dataset]
        self.adversary = LinfPGDAttack_mixed(self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2,
                                             nb_iter=30, eps_iter=0.02, rand_init=False, clip_min=-np.inf,
                                             clip_max=np.inf, targeted=False)

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.FloatTensor([funccall])
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, self.Dataset)
        return t_diagnosis_codes

    def classify(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes[0], weight_of_embed_codes[1])
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
                    greedy_set_best_temp_funccall, adversary):
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
        for i in range(0, min(len(greedy_set) + 1, budgets[self.Dataset])):
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
        funccall_lists = np.array(funccall_lists)
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
            batch_diagnosis_codes = torch.FloatTensor(funccall_lists[batch_size * index: batch_size * (index + 1)])
            batch_labels = torch.tensor([orig_label] * len(batch_diagnosis_codes)).cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)

            with ctx_noparamgrad_and_eval(self.model):
                t_diagnosis_codes[0] = adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                         batch_labels).detach()
                funccall_lists[batch_size * index: batch_size * (index + 1), :self.n_con_fea] = t_diagnosis_codes[
                    0].cpu().numpy()
            logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
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
                batch_labels = torch.tensor([list_label_set[i]] * len(batch_diagnosis_codes)).cuda()
                if t_diagnosis_codes[1].size(0) == 1:
                    flag = 1
                    # if Dataset_type[Dataset] == 'multi':
                    t_diagnosis_codes[1] = t_diagnosis_codes[1].repeat(2, 1, 1)
                    t_diagnosis_codes[0] = t_diagnosis_codes[0].repeat(2, 1)
                    # else:
                    #     t_diagnosis_codes[1] = t_diagnosis_codes.repeat(2, 1)
                    batch_labels = batch_labels.repeat(2)
                t_diagnosis_codes[1] = torch.autograd.Variable(t_diagnosis_codes[1].data, requires_grad=True)
                logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
                loss = self.criterion(logit, batch_labels)
                loss.backward()
                # we use the gradient of the false label. since there are only 3 lables, we just use grad_0 and _1
                grad = t_diagnosis_codes[1].grad.cpu().data
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
            # if Dataset_type[Dataset] == 'multi':
            grad_feature_temp = torch.max(grad, dim=2)[0]
            grad_feature_temp = grad_feature_temp / subsets_g
            grad_cate_index = torch.argmax(grad, dim=2)
            grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=0)
            # else:
            #     grad_feature_temp = grad / subsets_g
            grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=0)

        # if the one of the candidates attacks successfully, then we exit.
        if max_subsets_object >= 0 or len(greedy_set) == num_feature[self.Dataset]:
            if max_subsets_object >= 0:
                # print(max_subsets_object)
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
            if (index.item() + self.n_con_fea) in greedy_set_visit_idx:
                continue
            temp_funccall = copy.deepcopy(funccall_lists[grad_set_index_list[index]])
            # if Dataset_type[Dataset] == 'multi':
            if self.Dataset in complex_categories.keys():
                if grad_cate_index_list[grad_set_index_list[index], index] >= complex_categories[self.Dataset][index]:
                    continue
            temp_funccall[index + self.n_con_fea] = grad_cate_index_list[grad_set_index_list[index], index]
            # elif Dataset_type[Dataset] == 'binary':
            #     temp_funccall[index] = 1 - temp_funccall[index]
            features.append(index + self.n_con_fea)
            funccalls.append(temp_funccall)

        if not funccalls:
            success_flag = -2
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                greedy_set_visit_idx, flip_set, flip_funccall, query_num

        funccalls = torch.LongTensor(funccalls)
        query_num += len(features)
        t_diagnosis_codes = input_process(funccalls, self.Dataset)
        batch_labels = torch.tensor([orig_label] * len(funccalls)).cuda()
        with ctx_noparamgrad_and_eval(self.model):
            t_diagnosis_codes[0] = adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                     batch_labels).detach()
            funccalls = funccalls.numpy()
            funccalls[:, :self.n_con_fea] = t_diagnosis_codes[0].cpu().numpy()
        logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
        objects = h - g

        max_object = np.max(objects)
        max_index = np.argmax(objects)

        max_feature = features[max_index].item()
        # if Dataset_type[Dataset] == 'multi':
        max_category = grad_cate_index_list[
            grad_set_index_list[max_feature - self.n_con_fea], max_feature - self.n_con_fea].item()
        # elif Dataset_type[Dataset] == 'binary':
        #     max_category = int(1 - eval_funccall[max_feature])
        # else:
        #     max_category = None
        # if the max object changs, we update it and the best funccall
        if max_object < max_subsets_object:
            max_object = max_subsets_object
            greedy_set_best_temp_funccall = funccall_lists[max_subset_index]
        else:
            greedy_set_best_temp_funccall = funccalls[max_index]

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
        diff_set = set(np.where(eval_funccall[self.n_con_fea:] != new_funccall[self.n_con_fea:])[0])
        return diff_set

    def attack(self, funccall, y):
        # print()
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
        time_limit = OMPGS_time_limits[self.Dataset]

        # when the classification is wrong for the original example, skip the attack_utils
        if current_object > 0:
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
        # once the success_flag ==0, we attack_utils successfully and exit
        st = time.time()
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
                flip_set, flip_funccall, query_num = self.eval_object(funccall, greedy_set, y,
                                                                      greedy_set_visit_idx, query_num,
                                                                      greedy_set_best_temp_funccall, self.adversary)

            # print(iteration)
            # print(worst_object)
            # print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(worst_object)
            greedy_set_process.append(copy.deepcopy(greedy_set))

            # time limit exceed or we have attacked all the features, but it is still not successful.
            # if (time.time() - st) > time_limit or success_flag == -2:
            if iteration == budgets[self.Dataset] or success_flag == -2:
                success_flag = -1
                robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration

    def funccall_query(self, eval_funccall, greedy_set):
        candidate_lists = []
        funccall_lists = []

        for i in range(min(len(greedy_set) + 1, budgets[self.Dataset])):
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

    def attack_FSGS(self, funccall, y):
        time_limit = 1
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
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration
        # print(current_object)
        # once the success_flag ==0, we attack_utils successfully and exit
        st = time.time()
        while success_flag == 1:
            iteration += 1
            success_flag = 1
            pos_dict = {}
            funccall_lists_all = []

            # we loop over each feature and each category to find the worst object and its position
            for visit_idx in range(self.n_con_fea, len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                for code_idx in range(num_avail_category[self.Dataset]):
                    if code_idx == funccall[visit_idx]:
                        continue
                    if self.Dataset in complex_categories.keys():
                        if code_idx >= complex_categories[self.Dataset][visit_idx - self.n_con_fea]:
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

                batch_diagnosis_codes = torch.FloatTensor(
                    funccall_lists_all[batch_size * index: batch_size * (index + 1)])
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                with ctx_noparamgrad_and_eval(self.model):
                    t_diagnosis_codes[0] = self.adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                                  torch.LongTensor([y] * t_diagnosis_codes[0].shape[
                                                                      0]).cuda()).detach()
                logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
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
            greedy_set_best_temp_funccall = np.array(greedy_set_best_temp_funccall)
            with ctx_noparamgrad_and_eval(self.model):
                t_diagnosis_codes = self.input_handle(greedy_set_best_temp_funccall)
                t_diagnosis_codes[0] = self.adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                              torch.LongTensor([y]).cuda()).detach()
                greedy_set_best_temp_funccall[:self.n_con_fea] = t_diagnosis_codes[0].cpu().numpy()

            # print(iteration)
            # print('query', query_num)
            # print(max_object)

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

            # print(greedy_set)
            if success_flag == 1:
                # if (time.time() - st) > time_limit or len(greedy_set) == num_feature[self.Dataset]:
                if iteration == budgets[self.Dataset] or len(greedy_set) == num_feature[self.Dataset]:
                    success_flag = -1
                    robust_flag = 1
                    # print('Time out')

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)
            # print('Attack successfully')
        #
        # print("Modified_set:", flip_set)
        # print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration


def smooth_sampling(batch_data, Dataset, n=3):
    p_change = budgets[Dataset] / batch_data.size(1)
    batch_data_perturbed = torch.tensor([])
    for i in range(n):
        valid = torch.LongTensor(batch_data.shape).bernoulli_(p_change)
        if Dataset_type[Dataset] == 'binary':
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * (1 - batch_data)
        elif Dataset not in complex_categories.keys() and Dataset_type[Dataset] == 'multi':
            changed_to = np.argmax(np.random.multinomial(1, [1 / num_avail_category[Dataset]] *
                                                         num_avail_category[Dataset], size=batch_data.shape), axis=2)
            changed_to = torch.tensor(changed_to)
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * changed_to
        elif Dataset in complex_categories.keys() and Dataset_type[Dataset] == 'mixed':
            valid = torch.LongTensor(batch_data[:, num_con_feature[Dataset]:].shape).bernoulli_(p_change)
            changed_to = torch.tensor([])
            noise = torch.randn(batch_data.size(0), num_con_feature[Dataset]) * 0.15
            batch_data_perturbed_t_con = batch_data[:, :num_con_feature[Dataset]] + noise
            for j in range(num_feature[Dataset]):
                changed_to_t = np.argmax(np.random.multinomial(1, [1 / complex_categories[Dataset][j]] *
                                                               complex_categories[Dataset][j],
                                                               size=(batch_data.size(0), 1)), axis=2)
                changed_to_t = torch.tensor(changed_to_t)
                changed_to = torch.cat((changed_to, changed_to_t), dim=1)
            batch_data_perturbed_t_cat = batch_data[:, num_con_feature[Dataset]:] * (1 - valid) + valid * changed_to
            batch_data_perturbed_t = torch.cat((batch_data_perturbed_t_con, batch_data_perturbed_t_cat), dim=1)

        elif Dataset in complex_categories.keys() and Dataset_type[Dataset] == 'multi':
            changed_to = torch.tensor([])
            for j in range(batch_data.size(1)):
                changed_to_t = np.argmax(np.random.multinomial(1, [1 / complex_categories[Dataset][j]] *
                                                               complex_categories[Dataset][j],
                                                               size=(batch_data.size(0), 1)), axis=2)
                changed_to_t = torch.tensor(changed_to_t)
                changed_to = torch.cat((changed_to, changed_to_t), dim=1)
            batch_data_perturbed_t = batch_data * (1 - valid) + valid * changed_to
        else:
            batch_data_perturbed_t = None
        batch_data_perturbed = torch.cat((batch_data_perturbed, batch_data_perturbed_t), dim=0).float()
    return batch_data_perturbed


def IntegratedGradient(X_Train, y_Train, Dataset, n_batches_IG, invalid_sample, model):
    IG_matrix_all = torch.tensor(0)
    # model.apply(fix_bn)
    for k in range(20):
        IG_matrix = None
        for index in range(n_batches_IG):
            batch_diagnosis_codes = X_Train[512 * index: 512 * (index + 1)]
            t_labels = y_Train[512 * index: 512 * (index + 1)].cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            temp_codes = invalid_sample + (t_diagnosis_codes - invalid_sample) * (k / 20)
            temp_codes = torch.autograd.Variable(temp_codes.data, requires_grad=True)
            logit = model(temp_codes)
            loss = F.cross_entropy(logit, t_labels)
            # loss.backward()
            # temp_grad = temp_codes.grad.cpu().data
            temp_grad = torch.autograd.grad(loss, temp_codes, create_graph=True)[0].cpu().data
            if Dataset_type[Dataset] == 'multi':
                temp_IG_matrix = torch.sum(temp_grad * t_diagnosis_codes.cpu(), dim=2)
            elif Dataset_type[Dataset] == 'binary':
                temp_IG_matrix = temp_grad * t_diagnosis_codes.cpu()
            else:
                temp_IG_matrix = None
            if index == 0:
                IG_matrix = temp_IG_matrix
            else:
                IG_matrix = torch.cat((IG_matrix, temp_IG_matrix), dim=0)

        IG_matrix_all = IG_matrix + IG_matrix_all
    IG_matrix_all = IG_matrix_all / -20
    attributions_all = F.softmax(IG_matrix_all, dim=1)
    attributions = torch.mean(attributions_all, dim=0)
    return attributions, IG_matrix_all


def IntegratedGradient_Batch(X_Train, y_Train, Dataset, invalid_sample, model, steps=20, create_graph=True, softmax=True):
    IG_matrix_all = torch.tensor(0)
    for k in range(steps):
        IG_matrix = None
        t_labels = y_Train.cuda()
        t_diagnosis_codes = input_process(X_Train, Dataset)
        temp_codes = invalid_sample + (t_diagnosis_codes - invalid_sample) * (k / steps)
        temp_codes = torch.autograd.Variable(temp_codes.data, requires_grad=True)
        logit = model(temp_codes)
        loss = F.cross_entropy(logit, t_labels)
        temp_grad = torch.autograd.grad(loss, temp_codes, create_graph=create_graph)[0]
        if Dataset_type[Dataset] == 'multi':
            IG_matrix = torch.sum(temp_grad * t_diagnosis_codes, dim=2)
        elif Dataset_type[Dataset] == 'binary':
            IG_matrix = temp_grad * t_diagnosis_codes
        else:
            IG_matrix = None

        IG_matrix_all = IG_matrix + IG_matrix_all
        if not create_graph:
            torch.cuda.empty_cache()
            IG_matrix_all = IG_matrix_all.detach()
    IG_matrix_all = IG_matrix_all * -steps*10
    IG_matrix_all = abs(IG_matrix_all)
    attributions_all = IG_matrix_all
    if softmax:
        attributions_all = F.softmax(IG_matrix_all, dim=1)
    attributions = attributions_all
    return attributions, IG_matrix_all


def IntegratedGradient_mixed(X_Train, y_Train, Dataset, n_batches_IG, invalid_sample, model):
    model.train()
    model.apply(fix_bn)
    IG_matrix_all = torch.tensor(0)
    for k in range(20):
        IG_matrix = None
        for index in range(n_batches_IG):
            batch_diagnosis_codes = X_Train[512 * index: 512 * (index + 1)]
            t_labels = y_Train[512 * index: 512 * (index + 1)].cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

            temp_codes_con = invalid_sample[0] + (t_diagnosis_codes[0] - invalid_sample[0]) * (k / 20)
            temp_codes_cat = invalid_sample[1] + (t_diagnosis_codes[1] - invalid_sample[1]) * (k / 20)
            temp_codes_con = torch.autograd.Variable(temp_codes_con.data, requires_grad=True)
            temp_codes_cat = torch.autograd.Variable(temp_codes_cat.data, requires_grad=True)
            logit = model(temp_codes_con, temp_codes_cat)
            loss = F.cross_entropy(logit, t_labels)
            loss.backward()
            temp_con_grad = temp_codes_con.grad.cpu().data
            temp_cat_grad = temp_codes_cat.grad.cpu().data
            # if Dataset_type[Dataset] == 'multi':
            temp_IG_matrix_con = temp_con_grad * t_diagnosis_codes[0].cpu()
            temp_IG_matrix_cat = torch.sum(temp_cat_grad * t_diagnosis_codes[1].cpu(), dim=2)
            temp_IG_matrix = torch.cat((temp_IG_matrix_con, temp_IG_matrix_cat), dim=1)
            # else:
            #     temp_IG_matrix = None
            if index == 0:
                IG_matrix = temp_IG_matrix
            else:
                IG_matrix = torch.cat((IG_matrix, temp_IG_matrix), dim=0)

        IG_matrix_all = IG_matrix + IG_matrix_all
    IG_matrix_all = IG_matrix_all / -20

    attributions_all = IG_matrix_all / torch.norm(IG_matrix_all, p=2, dim=1, keepdim=True)
    attributions_all = F.softmax(attributions_all, dim=1)
    attributions = torch.mean(attributions_all, dim=0)
    model.apply(tune_bn)
    return attributions, IG_matrix_all


def IntegratedGradient_mixed_Batch(X_Train, y_Train, Dataset, invalid_sample, model, steps=20, create_graph=True, temp=1, softmax=True):
    model.apply(fix_bn)
    IG_matrix_all = torch.tensor(0)
    for k in range(steps):
        t_labels = y_Train.cuda()
        t_diagnosis_codes = input_process(X_Train, Dataset)
        temp_codes_con = invalid_sample[0] + (t_diagnosis_codes[0] - invalid_sample[0]) * (k / steps)
        temp_codes_cat = invalid_sample[1] + (t_diagnosis_codes[1] - invalid_sample[1]) * (k / steps)
        temp_codes_con = torch.autograd.Variable(temp_codes_con.data, requires_grad=True)
        temp_codes_cat = torch.autograd.Variable(temp_codes_cat.data, requires_grad=True)
        logit = model(temp_codes_con, temp_codes_cat)
        loss = F.cross_entropy(logit, t_labels)

        temp_con_grad = torch.autograd.grad(loss, temp_codes_con, create_graph=True)[0]
        temp_cat_grad = torch.autograd.grad(loss, temp_codes_cat, create_graph=create_graph)[0]

        temp_IG_matrix_con = temp_con_grad * t_diagnosis_codes[0]
        temp_IG_matrix_cat = torch.sum(temp_cat_grad * t_diagnosis_codes[1], dim=2)
        IG_matrix = torch.cat((temp_IG_matrix_con, temp_IG_matrix_cat), dim=1)
        IG_matrix = IG_matrix / temp

        IG_matrix_all = IG_matrix + IG_matrix_all
        if not create_graph:
            torch.cuda.empty_cache()
            IG_matrix_all = IG_matrix_all.detach()
    IG_matrix_all = IG_matrix_all / -steps
    # IG_matrix_all = abs(IG_matrix_all)
    # IG_matrix_all = F.normalize(IG_matrix_all, p=2, dim=1)

    attributions_all = IG_matrix_all
    if softmax:
        attributions_all = F.softmax(IG_matrix_all, dim=1)
    attributions = attributions_all
    model.apply(tune_bn)
    return attributions, IG_matrix_all


# load the dataset
def preparation(dataset):
    x = pickle.load(open('./dataset/' + dataset + 'X.pickle', 'rb'))
    y = pickle.load(open('./dataset/' + dataset + 'Y.pickle', 'rb'))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


def invalid_sample_(Dataset):
    if Dataset_type[Dataset] == 'multi':
        invalid_sample = torch.tensor([num_category[Dataset] - 1] * num_feature[Dataset]).unsqueeze(0)
        invalid_sample = one_hot_samples(invalid_sample, Dataset)[0].cuda()
    elif Dataset_type[Dataset] == 'binary':
        invalid_sample = torch.tensor([0] * num_feature[Dataset]).cuda()
    elif Dataset_type[Dataset] == 'mixed':
        invalid_sample_cat = torch.tensor([num_category[Dataset] - 1] * num_feature[Dataset]).unsqueeze(0)
        invalid_sample_cat = one_hot_samples(invalid_sample_cat, Dataset)[0].cuda()
        invalid_sample_con = torch.tensor([0] * num_con_feature[Dataset]).cuda()
        invalid_sample = [invalid_sample_con, invalid_sample_cat]
    else:
        invalid_sample = None
    return invalid_sample


def valid_mat_(Dataset, model):
    if Dataset in complex_categories.keys():
        valid_mat = model.valid_matrix().cuda()
    else:
        valid_mat = torch.ones(num_feature[Dataset], num_category[Dataset]).cuda()
        valid_mat[:, -1] = 0
    return valid_mat


def adjust_learning_rate(args, optimizer, epoch, n_epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch / n_epoch >= 0.6:
        lr = args.lr * 0.1
    elif epoch / n_epoch >= 0.85:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# MLP
alphas = {'Splice': 0.1, 'pedec': 0.01, 'census': 0.1}
betas = {'Splice': 0.01, 'pedec': 1, 'census': 0.01}
# Transformer
# alphas = {'Splice': 1, 'pedec': 0.1, 'census': 1}
# betas = {'Splice': 100, 'pedec': 100, 'census': 100}
gammas = {'Splice': 1, 'pedec': 1, 'census': 0.02}
deltas = {'Splice': 1, 'pedec': 0.05, 'census': 0.1}
epsilons = {'Splice': 0.00001, 'pedec': 0.00000001, 'census': 0.001}
epochs = {'Splice': 3000, 'pedec': 180, 'census': 100}
batch_sizes = {'Splice': 64, 'pedec': 128, 'census': 512}