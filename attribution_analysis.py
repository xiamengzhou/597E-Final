from utils import *

from torch.utils.data import (DataLoader)
from captum.attr import IntegratedGradients
import sys
import tqdm

model_name = sys.argv[1] # dnn, lr
split = sys.argv[2] # train, test
out_file = sys.argv[3]

print(model_name, split, out_file)
batch_size = 128

# get data
data = read_data(split)
loader = DataLoader(data, sampler=None, batch_size=batch_size, shuffle=False)
median_data = read_data("median")

# construct a baseline
baseline = median_data[0][0].unsqueeze(0).cuda()

# load model
model = load_pytorch_model(model_name)
model = model.cuda()

# initialize integrated gradients
ig = IntegratedGradients(model)

# get feature names
columns = get_columns()

# auc
acc = AccuracyStat()

attribution_sum = 0
for batch in tqdm.tqdm(loader):
    inputs = batch[0].cuda()
    out = model(inputs)
    acc.step(out, batch[1])
    attributions, approximation_error = ig.attribute(inputs, baselines=baseline, target=0, return_convergence_delta=True)
    attribution_sum += torch.sum(attributions, axis=0)

attribution_sum = attribution_sum.cpu().numpy() / len(loader)
sorted_attribution = sorted(enumerate(attribution_sum), key=lambda x: abs(x[1]))

with open(out_file, "w") as f:
    accuracy, auc = acc.get_accuracy()
    f.write(f"accuracy: {accuracy}, auc: {auc}")
    for index, attribution in sorted_attribution:
        f.write(f"{index}, {columns[index]}, {attribution}\n")

