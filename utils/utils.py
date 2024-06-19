import numpy as np
import torch
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from math import log
import copy
import torch.nn as nn
from advertorch.utils import replicate_input, is_float_or_torch_tensor, batch_multiply, batch_clamp, batch_l1_proj
from advertorch.attacks import Attack
from advertorch.attacks.utils import rand_init_delta, clamp, normalize_by_pnorm
import json


# if test=True, load the test file, or load the whole file
def load_data(Dataset, test=True):
    if test:
        test_idx = pickle.load(open(Test_Idx_File[Dataset], 'rb'))
        whole_data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
        whole_label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
        data = whole_data[test_idx]
        label = whole_label[test_idx]
        return data, label
    data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
    label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
    return data, label


def dataset_split(Dataset):
    train_idx = pickle.load(open('./dataset/' + Dataset + '_train_idx.pickle', 'rb'))
    test_idx = pickle.load(open('./dataset/' + Dataset + '_test_idx.pickle', 'rb'))
    return train_idx, test_idx


def store_time(dataset, Model_Name, training_time):
    if os.path.exists('./Logs/training_time.json'):
        training_times = json.load(open('./Logs/training_time.json', 'r'))
        training_times[dataset][Model_Name] = training_time
        f = open('./Logs/training_time.json', 'w')
        json.dump(training_times, f)
    else:
        training_times = {}
        training_times['census'] = {}
        training_times['Splice'] = {}
        training_times['pedec'] = {}
        training_times[dataset][Model_Name] = training_time
        f = open('./Logs/training_time.json', 'w')
        json.dump(training_times, f)


def get_decoder_pars(model):
    pars = []
    par_names = []
    for n, m in model.named_parameters():
        if 'encoder' in n:
            continue
        pars.append(m)
        par_names.append(n)

    return pars, par_names


# make sure the path exist
def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)


# make some vector one hot vector
def one_hot_labels(y, n_labels):
    return torch.zeros(y.size(0), n_labels).long().scatter(1, y.unsqueeze(1).cpu(), 1).cuda()


def one_hot_samples(x, dataset):
    return torch.zeros(x.size(0), x.size(1), num_category[dataset]).long().scatter(2, x.unsqueeze(2).long().cpu(), 1)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def tune_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def input_process(batch_diagnosis_codes, Dataset):
    if Dataset_type[Dataset] == 'multi':
        t_diagnosis_codes = one_hot_samples(batch_diagnosis_codes, Dataset).cuda().float()
    elif Dataset_type[Dataset] == 'mixed':
        t_diagnosis_codes_cat = one_hot_samples(batch_diagnosis_codes[:, num_con_feature[Dataset]:],
                                                Dataset).cuda().float()
        t_diagnosis_codes = [batch_diagnosis_codes[:, :num_con_feature[Dataset]].float().cuda(), t_diagnosis_codes_cat]
    else:
        t_diagnosis_codes = batch_diagnosis_codes.cuda().float()
    return t_diagnosis_codes


class LabelMixin(object):
    def _get_predicted_label(self, x_con, x_cat):
        with torch.no_grad():
            outputs = self.predict(x_con, x_cat)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x_con, x_cat, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x_con, x_cat)

        x_con = replicate_input(x_con)
        x_cat = replicate_input(x_cat)
        y = replicate_input(y)
        return x_con, x_cat, y


class PGDAttack(Attack, LabelMixin):
    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False):
        """
        Create an instance of the PGDAttack.

        """
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)


class LinfPGDAttack_mixed(PGDAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack_utils step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack_utils is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.05, nb_iter=40,
            eps_iter=0.005, rand_init=True, clip_min=-100, clip_max=100,
            targeted=False):
        ord = np.inf
        super(LinfPGDAttack_mixed, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)

    def perturb_iterative(self, xvar, x_cat, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                          delta_init=None, minimize=False, ord=np.inf,
                          clip_min=0.0, clip_max=1.0,
                          l1_sparsity=None):
        if delta_init is not None:
            delta = delta_init
        else:
            delta = torch.zeros_like(xvar)

        delta.requires_grad_()
        for ii in range(nb_iter):
            outputs = predict(xvar + delta, x_cat)
            loss = loss_fn(outputs, yvar)
            if minimize:
                loss = -loss

            loss.backward()
            if ord == np.inf:
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
                delta.data = batch_clamp(eps, delta.data)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                   ) - xvar.data

            else:
                error = "Only ord = inf have been implemented"
                raise NotImplementedError(error)
            delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, clip_min, clip_max)
        return x_adv

    def perturb(self, x_con, x_cat, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack_utils length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x_con, x_cat, y = self._verify_and_process_inputs(x_con, x_cat, y)

        delta = torch.zeros_like(x_con)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x_con, self.ord, 0.25 * self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x_con + delta.data, min=self.clip_min, max=self.clip_max) - x_con

        rval = self.perturb_iterative(
            x_con, x_cat, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval


class LmixPGDAttack_mixed(PGDAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack_utils step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack_utils is targeted.
    """

    def __init__(
            self, predict, loss_fn=None, eps_con=0.05, eps_cat=5, nb_iter=40,
            eps_iter_con=0.005, eps_iter_cat=0.2, rand_init=True, clip_min=-100, clip_max=100,
            targeted=False):
        ord = np.inf
        super(LmixPGDAttack_mixed, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps_con, nb_iter=nb_iter,
            eps_iter=eps_iter_con, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord)

        self.eps_con = eps_con
        self.eps_cat = eps_cat
        self.eps_iter_con = eps_iter_con
        self.eps_iter_cat = eps_iter_cat

    def perturb_iterative(self, xvar, x_cat, yvar, predict, nb_iter, eps_con, eps_cat, eps_iter_con, eps_iter_cat,
                          loss_fn, delta_init_con=None, delta_init_cat=None, minimize=False, ord=np.inf,
                          clip_min=0.0, clip_max=1.0):
        if delta_init_con is not None:
            delta_con = delta_init_con
        else:
            delta_con = torch.zeros_like(xvar)
        if delta_init_cat is not None:
            delta_cat = delta_init_cat
        else:
            delta_cat = torch.zeros_like(x_cat)

        delta_con.requires_grad_()
        delta_cat.requires_grad_()
        for ii in range(nb_iter):
            outputs = predict(xvar + delta_con, x_cat + delta_cat)
            loss = loss_fn(outputs, yvar)
            if minimize:
                loss = -loss

            loss.backward()
            grad_sign_con = delta_con.grad.data.sign()
            delta_con.data = delta_con.data + batch_multiply(eps_iter_con, grad_sign_con)
            delta_con.data = batch_clamp(eps_con, delta_con.data)
            delta_con.data = clamp(xvar.data + delta_con.data) - xvar.data

            grad_cat = delta_cat.grad.data
            abs_grad = torch.abs(grad_cat)

            batch_size = grad_cat.size(0)
            view = abs_grad.view(batch_size, -1)
            vals, idx = view.topk(1)

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad_cat)
            grad_cat = grad_cat.sign() * (out > 0).float()
            grad_cat = normalize_by_pnorm(grad_cat, p=1)
            delta_cat.data = delta_cat.data + batch_multiply(eps_iter_cat, grad_cat)

            delta_cat.data = batch_l1_proj(delta_cat.data.cpu(), eps_cat)
            if x_cat.is_cuda:
                delta_cat.data = delta_cat.data.cuda()
            delta_cat.data = clamp(x_cat.data + delta_cat.data, clip_min, clip_max
                                   ) - x_cat.data
            delta_cat.grad.data.zero_()

        x_adv_con = clamp(xvar + delta_con)
        x_adv_cat = clamp(x_cat + delta_cat, clip_min, clip_max)
        return x_adv_con, x_adv_cat

    def perturb(self, x_con, x_cat, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack_utils length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x_con, x_cat, y = self._verify_and_process_inputs(x_con, x_cat, y)

        delta_con = torch.zeros_like(x_con)
        delta_con = nn.Parameter(delta_con)
        delta_cat = torch.zeros_like(x_cat)
        delta_cat = nn.Parameter(delta_cat)
        if self.rand_init:
            rand_init_delta(
                delta_con, x_con, np.inf, self.eps_con, -np.inf, np.inf)
            delta_con.data = clamp(
                x_con + delta_con.data) - x_con
            rand_init_delta(
                delta_cat, x_cat, 1, self.eps_cat, self.clip_min, self.clip_max)
            delta_con.data = clamp(
                x_con + delta_con.data, min=self.clip_min, max=self.clip_max) - x_con

        rval = self.perturb_iterative(
            x_con, x_cat, y, self.predict, nb_iter=self.nb_iter,
            eps_con=self.eps_con, eps_cat=self.eps_cat, eps_iter_con=self.eps_iter_con, eps_iter_cat=self.eps_iter_cat,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init_con=delta_con, delta_init_cat=delta_cat)

        return rval


Test_Idx_File = {
    'Splice': './dataset/Splice_test_idx.pickle',
    'pedec': './dataset/pedec_test_idx.pickle',
    'census': './dataset/census_test_idx.pickle',
}

Train_Idx_File = {
    'Splice': './dataset/Splice_train_idx.pickle',
    'pedec': './dataset/pedec_train_idx.pickle',
    'census': './dataset/census_train_idx.pickle',
}

Whole_Data_File = {
    'Splice': './dataset/SpliceX.pickle',
    'pedec': './dataset/pedecX.pickle',
    'census': './dataset/censusX.pickle',
}

Whole_Label_File = {
    'Splice': './dataset/SpliceY.pickle',
    'pedec': './dataset/pedecY.pickle',
    'census': './dataset/censusY.pickle',
}

num_category = {'Splice': 5, 'pedec': 3, 'census': 52}
num_feature = {'Splice': 60, 'pedec': 5000, 'census': 32}
num_samples = {'Splice': 3190, 'pedec': 21790, 'census': 299285}
num_avail_category = {'Splice': 4, 'pedec': 2, 'census': 51}
num_classes = {'Splice': 3, 'pedec': 2, 'census': 2}
num_con_feature = {'census': 9, 'Splice': 0, 'pedec': 0}
forbid_feaures = {'Splice': [28, 29, 30, 31, 32], 'pedec': [326, 3664, 3861, 4865, 4390], 'census': [4, 5, 0, 8, 1]}

census_category = [9, 17, 3, 7, 24, 15, 5, 10, 2, 3, 6, 8, 6, 6, 51, 38,
                   8, 10, 9, 10, 3, 4, 7, 5, 43, 43, 43, 5, 3, 3, 3, 2]

complex_categories = {
    'census': census_category
}


Dataset_type = {'Splice': 'multi',
                'pedec': 'multi',
                'census': 'mixed',
                }

budgets = {'Splice': 5, 'pedec': 5, 'census': 5}
OMPGS_time_limits = {'Splice': 1, 'pedec': 2, 'census': 1.2}
FSGS_time_limits = {'Splice': 1, 'pedec': 150, 'census': 2}
PCAA_time_limits = {'Splice': 1, 'pedec': 2, 'census': 1.2}

