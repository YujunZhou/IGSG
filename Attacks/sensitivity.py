import argparse
from filelock import FileLock
from utils.Training_utils import *

# creating parser object
parser = argparse.ArgumentParser(description='OMPGS')
# parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='census', type=str, help='dataset')
parser.add_argument('--modeltype', default='MLP_Normal', type=str, help='model type')
parser.add_argument('--idx', default=0, type=int, help='running index')
# parser.add_argument('--time', default=3, type=int, help='time limit')
parser.add_argument('--t', default='True', type=str, help='test set or whole set')
args = parser.parse_args()


def sensitivity_analysis(X, y, n_labels, n_cat):
    batch_size = 512
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    sensitivity = torch.tensor(0)
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        t_labels = y[batch_size * index: batch_size * (index + 1)].cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)

        if Dataset != 'census':
            logit = model(t_diagnosis_codes)
        else:
            logit = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
        y_para = one_hot_labels(t_labels, n_labels)
        y_prob = logit * y_para
        y_prob = torch.max(y_prob, 1)[0].cpu()

        decrease = torch.tensor([])

        for i in range(num_con_feature[Dataset], num_con_feature[Dataset] + num_feature[Dataset]):
            batch_diagnosis_codes_temp = copy.deepcopy(batch_diagnosis_codes)
            decrease_temp = torch.tensor([])
            for j in range(n_cat):
                if Dataset in complex_categories.keys():
                    if j >= complex_categories[Dataset][i-num_con_feature[Dataset]]:
                        break
                for k in range(len(batch_diagnosis_codes)):
                    batch_diagnosis_codes_temp[k][i] = j
                with torch.no_grad():
                    t_diagnosis_codes_temp = input_process(batch_diagnosis_codes_temp, Dataset)
                    if Dataset != 'census':
                        logit = model(t_diagnosis_codes_temp)
                    else:
                        logit = model(t_diagnosis_codes_temp[0], t_diagnosis_codes_temp[1])
                    y_prob_temp = logit * y_para
                y_prob_temp = torch.max(y_prob_temp, 1)[0].cpu()
                y_prob_diff = y_prob - y_prob_temp
                y_prob_diff = y_prob_diff.relu()
                decrease_temp = torch.cat((decrease_temp, y_prob_diff.unsqueeze(0)), dim=0)
                torch.cuda.empty_cache()
            decrease_temp = torch.max(decrease_temp, dim=0)[0]
            # feature_sensitivity[i][batch_size * index: batch_size * (index + 1)] = decrease_temp
            decrease_temp = torch.sum(decrease_temp)
            decrease = torch.cat((decrease, decrease_temp.unsqueeze(0)), dim=0)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        sensitivity = sensitivity + decrease
    sensitivity = sensitivity / len(X)
    order = torch.argsort(sensitivity, descending=True).numpy()

    return sensitivity, order

Datasets = ['pedec', 'census', 'Splice']
archs = ['MLP', 'Transformer']
victim_models = ['Normal', 'PGD', 'FastBAT', 'TRADES', 'AFD', 'PCAA', 'IGR', 'JR', 'IGSG']

Dataset = args.dataset

train_idx, test_idx = dataset_split(Dataset)
X, y = preparation(Dataset)
y_Test = y[test_idx]
X_Test = X[test_idx]
if Dataset == 'Splice':
    from Models.SpliceModels import *
elif Dataset == 'pedec':
    from Models.PEDecModels import *
elif Dataset == 'census':
    from Models.CensusModels import *
Model_Type = args.modeltype
budget = budgets[Dataset]
time_limit = OMPGS_time_limits[Dataset]
t = True
best_parameters_file = './classifier/{}_{}_{}.par'.format(Dataset, Model_Type, args.idx)

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

# load the trained parameters of the classifier
model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
# We only test data, so use this
model.eval()

sensitivity, sensitivity_order = sensitivity_analysis(X_Test, y_Test, num_classes[Dataset], num_avail_category[Dataset])


lock = FileLock("./Logs/sensitivity.json.lock")
with lock:
    if os.path.exists('./Logs/sensitivity.json'):
        Sens = json.load(open('./Logs/sensitivity.json', 'r'))
    else:
        Sens = {'Splice': {}, 'pedec': {}, 'census': {}}

    if Model_Type in Sens[Dataset].keys():
        pass
    else:
        Sens[Dataset][Model_Type] = {}

    if 'sensitivity' in Sens[Dataset][Model_Type].keys():
        pass
    else:
        Sens[Dataset][Model_Type]['sensitivity'] = {}
        Sens[Dataset][Model_Type]['sensitivity_order'] = {}
    Sens[Dataset][Model_Type]['sensitivity'][str(args.idx)] = sensitivity.tolist()
    Sens[Dataset][Model_Type]['sensitivity_order'][str(args.idx)] = sensitivity_order.tolist()

    if list(Sens[Dataset][Model_Type]['sensitivity'].keys()) == [str(i) for i in range(5)]:
        Sens[Dataset][Model_Type]['sensitivity']['mean'] = np.mean(np.array([Sens[Dataset][Model_Type]['sensitivity'][str(i)] for i in range(5)]), axis=0).tolist()
        Sens[Dataset][Model_Type]['sensitivity_order']['mean'] = np.mean(np.array([Sens[Dataset][Model_Type]['sensitivity_order'][str(i)] for i in range(5)]), axis=0).tolist()
    json.dump(Sens, open('./Logs/sensitivity.json', 'w'))








