import pickle as pkl
import h5py
import numpy as np
import torch
from dnn import Model

def load_weights():
    weights = pkl.load(open("/Users/mengzhouxia/dongdong/Princeton/Courses/COS597E/Project/admissionprediction/models/dnn.pkl", "rb"))
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

def test_difference():
    input = torch.randn(4, 1060)
    model = Model()
    print(model(input))

    load_model(model)
    print(model(input))


def save_model():
    model = Model()
    load_model(model)
    torch.save(model, "/Users/mengzhouxia/dongdong/Princeton/Courses/COS597E/Project/admissionprediction/models/dnn.pytorch")

