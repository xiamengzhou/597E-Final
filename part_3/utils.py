import pickle as pkl
import torch
from dnn import Model
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
from torch.utils.data import TensorDataset

def load_weights():
    weights = pkl.load(open("/Users/mengzhouxia/dongdong/Princeton/Courses/COS597E/Project/admissionprediction/models/dnn.pkl", "rb"))
    print(weights.keys())
    return weights

def load_lr_weights():
    weights = pkl.load(open("/Users/mengzhouxia/dongdong/Princeton/Courses/COS597E/Project/admissionprediction/models/final_lr.pkl", "rb"))
    print(weights.keys())
    return weights

def wrap_parameters(weight, transpose=False):
    tensor = torch.tensor(weight)
    if transpose:
        tensor = torch.transpose(tensor, 0, 1)
    return torch.nn.parameter.Parameter(tensor)

def load_model(model):
    weights = load_weights()
    with torch.no_grad():
        model.dense4.weight = wrap_parameters(weights["dense_4_weight"], True)
        model.dense4.bias = wrap_parameters(weights["dense_4_bias"], False)
        model.dense3.weight = wrap_parameters(weights["dense_3_weight"], True)
        model.dense3.bias = wrap_parameters(weights["dense_3_bias"], False)
        model.dense2.weight = wrap_parameters(weights["dense_2_weight"], True)
        model.dense2.bias = wrap_parameters(weights["dense_2_bias"], False)
        model.dense1.weight = wrap_parameters(weights["dense_1_weight"], True)
        model.dense1.bias = wrap_parameters(weights["dense_1_bias"], False)

def load_pytorch_model(model):
    model = torch.load(f"models/{model}.pytorch")
    return model

def get_columns():
    data = pd.read_csv("data/impute_x_test.csv", "r")
    columns = data.columns
    return columns


def load_lr_model(model):
    weights = load_lr_weights()
    with torch.no_grad():
        model.dense.weight = wrap_parameters(weights["weight"], True)
        model.dense.bias = wrap_parameters(weights["bias"], False)

def test_difference():
    input = torch.randn(4, 1060)
    model = Model()
    print(model(input))

    load_model(model)
    print(model(input))


def save_model(model, name):
    torch.save(model, f"/Users/mengzhouxia/dongdong/Princeton/Courses/COS597E/Project/admissionprediction/models/{name}")

class AccuracyStat():
    def __init__(self):
        self.num = 0
        self.correct_num = 0

        self.preds = []
        self.labels = []

    def step(self, logits, labels):
        logits = logits.view(logits.shape[0])
        labels = labels.cpu().numpy()
        predicted_labels = logits.detach().cpu().numpy()
        self.num += len(labels)
        self.correct_num += sum(labels == (predicted_labels > 0.5))
        self.preds.extend(list(predicted_labels))
        self.labels.extend(list(labels))

    def get_accuracy(self):
        accuracy = self.correct_num / self.num
        auc = roc_auc_score(self.labels, self.preds)
        return accuracy, auc


def read_data(split="train", data_dir="data"):
    pt_name = os.path.join(data_dir, f"impute_x_{split}.pt")

    if os.path.isfile(pt_name):
        with open(pt_name, 'rb') as f:
            data = torch.load(f)
    else:
        file_name = f"{data_dir}/impute_x_{split}.csv"
        features = pd.read_csv(file_name, index_col=0)

        labels = open(f"data/y_{split}.tsv", "r").readlines()[1:]
        labels = [float(label.split("\t")[1]) for label in labels]

        features = torch.tensor(features.values, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        data = TensorDataset(features, labels)

        if not os.path.exists(os.path.dirname(pt_name)):
            os.makedirs(os.path.dirname(pt_name))

        with open(pt_name, 'wb') as f:
            torch.save(data, f)

    print("  Num %s examples = %d", split, len(data))
    return data


def prepare_step_inputs(input, median, step=40):
    values = []
    for v, v_b in zip(input, median):
        distance = (v_b - v) / step
        if distance == 0:
            vs = torch.ones(40) * v
        else:
            vs = torch.arange(v, v_b, distance, requires_grad=True)[:step]
        values.append(vs)
    input = torch.transpose(torch.stack(values), 0, 1)
    return input


