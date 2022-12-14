# code_docstring_relevance_checker
  A tiny transformer + LGBM based model to find relevance match between code and docstring

## Dataset:
  Python corpus was used for training the model with 70/30 split.

## Metric:
  F1-score was used for evaluation and AUC was monitored with FP/FN to avoid overfitting of the model.

## Emedding
  Explored different embedding but had to manage trade off between inference time and model performance
  We have used sentence-transformer to get better code and docstring representation.

## Process
Reason to go with LGBM: 
<ol>
<li>Light weight and less memory consumption </li>
<li>Feature interpretation </li>
<li>Faster training time </li>
</ol>
