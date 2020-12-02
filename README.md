# 597E-Final

## models
`dnn.model`: final neural network model from R <br /> <br />
`final_lr.model`: final logistic regression model from R <br /> <br />
`dnn.pkl`: weights of final neural network model <br /> <br />
`final_lr.pkl`: weights of final logistic regression model <br /> <br />
`dnn.pytorch`: final neural network model in PyTorch <br /> <br />
 
The PyTorch version of the neural network model can be accessed by 

```
import torch
model = torch.load("model/dnn.pytorch")
```

The structure of the model can be found in `dnn.py`.
