# 597E-Final: Hospital Admission at Emergency Department Triage Using Machine Learning

This repository provides the code for our final project for the Fall 2020 COS597E/SOC555 Limits to Prediction seminar at Princeton University. 

## Reproducing our report
- Section 4.1 Do misclassified samples have something in common?: <br />
  (Python) `part1/misclassified-samples-analysis.ipynb` (use with `misclassified_samples_analysis.tar.gz`) <br /> <br />
- Section 4.2: How sensitive is each classifier to data perturbations? <br />
  (R) `part2/sensitivityanalysis_dnn.R`, `part2/sensitivityanalysis_lr.R` and `part2/sensitivityanalysis_xgboost.R` (use with `sensitivity_analysis.zip`) <br /> <br />
- Section 4.3: How can we interpret the decisions of each classifier? <br />
  (Python) `part3/attribution_analysis.py ${split} ${model} ${outfile}` <br /> <br />
- Section 4.4: Are the trained classifiers generalizable to hospitals not represented in the training data? <br />
  (Python) `part4/cross_hospital_generalization_analysis.ipynb` (use with `x_train.csv` and `y_train.csv`) <br /> <br />
Please find the large files in the [Google Driver](https://drive.google.com/drive/folders/1mr3X5v_qCOysY83YUcR_vS25g3y-TI9U?usp=sharing). The code and data for the original paper (Hong et al. 2018) can be found at https://github.com/yaleemmlc/admissionprediction.


## Models
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
