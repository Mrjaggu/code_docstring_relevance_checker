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

### 🛠 Languages & Tools Used:

<p align="left">  
  <a href="https://www.python.org/" target="_blank"> <img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> </a> 
  <a href="https://git-scm.com/" target="_blank"> <img src="https://img.shields.io/badge/Git-282C34?logo=git" alt="Git logo" title="Git" height="25" /> </a> 
  <a href="https://jupyter.org/" target="_blank"> <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> </a> 
 
