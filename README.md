# code docstring relevance checker
  A tiny sentence transformer + LGBM based model to find relevance match between code and docstring

## Dataset:
  Python corpus was used for training the model with 70/30 split.

## Metric:
  F1-score was used for evaluation and AUC was monitored with FP/FN to avoid overfitting of the model.

## Emedding
  <ol><li>Explored different embedding but had to manage trade off between inference time and model performance</li>
  <li>We have used sentence-transformer to get better code and docstring representation.</li></ol>

## Process
Reason to go with LGBM: 
<ol>
<li>Light weight and less memory consumption </li>
<li>Feature interpretation </li>
<li>Faster training time </li>
</ol>
