from filelock import FileLock
import argparse
from utils.Training_utils import *

# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--dataset', default='pedec', type=str, help='Dataset (Splice, PEDec)')
parser.add_argument('--modeltype', default='MLP_Normal', type=str, help='MLP, CNN')
parser.add_argument('--alpha', default=0, type=float, help='setting of alpha')
parser.add_argument('--beta', default=0, type=float, help='setting of beta')
parser.add_argument('--idx', default=0, type=int, help='running index')
args = parser.parse_args()


def shift_tensor(input_tensor, n):
    if input_tensor.dim() != 1:
        raise ValueError("input_tensor needs to be a 1D tensor")
    if n < 0 or n > input_tensor.size(0):
        raise ValueError("n must be in the range [0, input_tensor.size(0)]")
    if n == 0:
        return input_tensor
    last_n = input_tensor[-n:]
    rest = input_tensor[:-n]
    return torch.cat((last_n, rest), 0)


# There are two datasets, some models have the same name for the two dataset, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Models.SpliceModels import *
elif args.dataset == 'pedec':
    from Models.PEDecModels import *
elif args.dataset == 'census':
    from Models.CensusModels import *
else:
    raise NotImplementedError('Dataset not recognized ({args.dataset})')

Dataset = args.dataset
sm_num = 5
Model_Type = args.modeltype
batch_size = 512
train_idx, test_idx = dataset_split(Dataset)
X, y = preparation(Dataset)
y_Test = y[test_idx]
X_Test = X[test_idx]
os.makedirs('./Logs/' + Dataset + '/' + args.modeltype + '/', exist_ok=True)

if Dataset != 'census':
    if 'Transformer' in Model_Type:
        if 'AFD' in Model_Type:
            E, Dc = Transformer_E(), Transformer_Dc()
            model = nn.Sequential(E, Dc)
        else:
            model = Transformer()
    else:
        if 'AFD' in Model_Type:
            E, Dc = MLP_E(), MLP_Dc()
            model = nn.Sequential(E, Dc)
        else:
            model = MLP()
else:
    if 'Transformer' in Model_Type:
        if 'AFD' in Model_Type:
            E = Transformer_E()
            model = Transformer_Dc(E)
        else:
            model = Transformer()
    else:
        if 'AFD' in Model_Type:
            E = MLP_E()
            model = MLP_Dc(E)
        else:
            model = MLP()
if torch.cuda.is_available():
    model = model.cuda()

CEloss = torch.nn.CrossEntropyLoss().cuda()
n_batches = int(np.ceil(float(len(X_Test)) / float(batch_size)))
# defines the invalid sample x' using in Integrated Gradient
invalid_sample = invalid_sample_(Dataset)



for args.idx in range(5):
    st = time.time()
    model.load_state_dict(torch.load('./classifier/{}_{}_{}.par'.format(Dataset, Model_Type, str(args.idx)), map_location='cpu'))
    model.train()
    model.apply(fix_bn)
    attributions = torch.tensor([])
    if Dataset != 'census':
        IG_matrix_all = torch.tensor([0.0] * num_feature[Dataset]).cuda()
    else:
        IG_matrix_all = torch.tensor([0.0] * (num_feature[Dataset] + num_con_feature[Dataset])).cuda()
    # find the valid categories for each feature
    try:
        valid_mat = valid_mat_(Dataset, model)
    except:
        temp_model = MLP()
        valid_mat = valid_mat_(Dataset, temp_model)

    training_st = time.time()

    clean_loss_sum = 0
    grad_loss_sum = 0
    attr_loss_sum = 0
    model.train()
    train_loss = 0
    for index in range(n_batches):
        # make X like one hot vectors.
        batch_diagnosis_codes = X_Test[batch_size * index: batch_size * (index + 1)]
        t_labels = y_Test[batch_size * index: batch_size * (index + 1)].cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        if Dataset != 'census':
            t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
        else:
            t_diagnosis_codes[0] = torch.autograd.Variable(t_diagnosis_codes[0].data, requires_grad=True)
            t_diagnosis_codes[1] = torch.autograd.Variable(t_diagnosis_codes[1].data, requires_grad=True)
        grad_loss = torch.tensor(0)
        attr_loss = torch.tensor(0)

        if Dataset != 'census':
            batch_attribution, _ = IntegratedGradient_Batch(batch_diagnosis_codes, t_labels, Dataset,
                                                                        invalid_sample, model, create_graph=False, softmax=True)
        else:
            batch_attribution, _ = IntegratedGradient_mixed_Batch(batch_diagnosis_codes, t_labels, Dataset,
                                           invalid_sample, model, create_graph=False, softmax=True)
        mean_attribution = torch.mean(batch_attribution, dim=0)
        attr_loss = 0
        for i in range(1, num_feature[Dataset]):
            attr_loss = torch.norm(mean_attribution - shift_tensor(mean_attribution, i), p=1) + attr_loss
        attr_loss = attr_loss / num_feature[Dataset]

        batch_diagnosis_codes_sm = smooth_sampling(batch_diagnosis_codes, Dataset, sm_num)
        t_labels_sm = y_Test[batch_size * index: batch_size * (index + 1)].repeat(sm_num).cuda()
        t_diagnosis_codes_sm = input_process(batch_diagnosis_codes_sm, Dataset)
        if Dataset != 'census':
            t_diagnosis_codes_sm = torch.autograd.Variable(t_diagnosis_codes_sm.data, requires_grad=True)
            logit = model(t_diagnosis_codes_sm)
        else:
            t_diagnosis_codes_sm[0] = torch.autograd.Variable(t_diagnosis_codes_sm[0].data, requires_grad=True)
            t_diagnosis_codes_sm[1] = torch.autograd.Variable(t_diagnosis_codes_sm[1].data, requires_grad=True)
            logit = model(t_diagnosis_codes_sm[0], t_diagnosis_codes_sm[1])

        loss = CEloss(logit, t_labels_sm)
        loss.backward(retain_graph=False)
        if Dataset != 'census':
            grad_sm = t_diagnosis_codes_sm.grad.data
            grad_orig_cate = torch.sum(grad_sm * t_diagnosis_codes_sm, dim=2, keepdim=True).repeat(1, 1,
                                                                                                   num_category[
                                                                                                       Dataset])
            grad_orig_cate = grad_orig_cate * valid_mat
            grad_sm = grad_sm - grad_orig_cate
            grad_norm = torch.norm(grad_sm) / np.sqrt(sm_num)
        else:
            grad_sm_cat = t_diagnosis_codes_sm[1].grad
            grad_sm_con = t_diagnosis_codes_sm[0].grad
            grad_orig_cate = torch.sum(grad_sm_cat * t_diagnosis_codes_sm[1], dim=2, keepdim=True).repeat(1, 1,
                                                                                                          num_category[
                                                                                                              Dataset])
            grad_orig_cate = grad_orig_cate * valid_mat
            grad_sm_cat = grad_sm_cat - grad_orig_cate
            grad_norm = torch.norm(grad_sm_cat) / np.sqrt(sm_num) + torch.norm(grad_sm_con) / np.sqrt(sm_num)

        grad_loss = grad_norm
        grad_loss_sum += grad_loss.item()
        attr_loss_sum += attr_loss.item()
        IG_matrix_all += mean_attribution

    IG_matrix_all = IG_matrix_all / n_batches
    IG_matrix_all = IG_matrix_all.detach().cpu().numpy()
    # set the num type of IG_matrix_all to float
    IG_matrix_all = [float(i) for i in IG_matrix_all]
    cal_time = time.time() - st
    print(time.time() - st)


    lock = FileLock("./Logs/%s/IGSG.json.lock" % Dataset)
    with lock:
        if os.path.exists('./Logs/%s/IGSG.json' % Dataset):
            igsg = json.load(open('./Logs/%s/IGSG.json' % Dataset, 'r'))
        else:
            igsg = {}

        if Model_Type in igsg.keys():
            pass
        else:
            igsg[Model_Type] = {}

        if 'IG' in igsg[Model_Type].keys():
            pass
        else:
            igsg[Model_Type]['time'] = {}
            igsg[Model_Type]['IG'] = {}
            igsg[Model_Type]['SG'] = {}
            igsg[Model_Type]['IG_value'] = {}
        igsg[Model_Type]['IG'][str(args.idx)] = float(attr_loss_sum / n_batches)
        igsg[Model_Type]['SG'][str(args.idx)] = float(grad_loss_sum / n_batches)
        igsg[Model_Type]['IG_value'][str(args.idx)] = IG_matrix_all
        igsg[Model_Type]['time'][str(args.idx)] = cal_time
        if list(igsg[Model_Type]['IG'].keys()) == [str(i) for i in range(5)]:
            igsg[Model_Type]['IG']['mean'] = np.mean([igsg[Model_Type]['IG'][str(i)] for i in range(5)])
            igsg[Model_Type]['IG']['std'] = np.std([igsg[Model_Type]['IG'][str(i)] for i in range(5)])
            igsg[Model_Type]['SG']['mean'] = np.mean([igsg[Model_Type]['SG'][str(i)] for i in range(5)])
            igsg[Model_Type]['SG']['std'] = np.std([igsg[Model_Type]['SG'][str(i)] for i in range(5)])
            igsg[Model_Type]['IG_value']['mean'] = np.mean([igsg[Model_Type]['IG_value'][str(i)] for i in range(5)],
                                                           axis=0).tolist()
            igsg[Model_Type]['IG_value']['std'] = np.std([igsg[Model_Type]['IG_value'][str(i)] for i in range(5)],
                                                            axis=0).tolist()
            igsg[Model_Type]['time']['mean'] = np.mean([igsg[Model_Type]['time'][str(i)] for i in range(5)])
        json.dump(igsg, open('./Logs/%s/IGSG.json' % Dataset, 'w+'))




