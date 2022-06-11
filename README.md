# SsciBERT: A pretrained language model for social scientific text

<img src="/logo.png" alt="logo" style="zoom: 67%;" />

## Introduction

The research for social science texts needs the support natural language processing tools. 

The pre-trained language model has greatly improved the accuracy of text mining in general texts. At present, there is an urgent need for a pre-trained language model specifically for the automatic processing of scientific texts in social science. 

We used the abstract of social science research as the training set. Based on the deep language model framework of BERT, we constructed [SSCI-BERT and SSCI-SciBERT](https://github.com/S-T-Full-Text-Knowledge-Mining/SSCI-BERT)  pre-training language models by [transformers/run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py). 

We designed four downstream tasks of  Text Classification on different social scientific article corpus to verify the performance of the model.

- SSCI-BERT and SSCI-SciBERT are trained on the abstract of  articles published in SSCI journals from 1986 to 2021. The training set involved in the experiment included a total of `503910614 words`. 
- Based on the idea of Domain-Adaptive Pretraining, `SSCI-BERT` and `SSCI-SciBERT` combine a large amount of abstracts of scientific articles  based on the BERT structure, and continue to train the BERT and SSCI-SciBERT models respectively to obtain pre-training models for the automatic processing of Social science research texts. 



## News 

- 2022-03-24 : SSCI-BERT and SSCI-SciBERT has been put forward for the first time.
- 2022-06-09 : The paper for SsciBERT has been submitted to arxiv(https://arxiv.org/abs/2206.04510).



##  How to use

### Huggingface Transformers 

The `from_pretrained` method based on [Huggingface Transformers](https://github.com/huggingface/transformers) can directly obtain SSCI-BERT and SSCI-SciBERT models online. 



- SSCI-BERT

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/SSCI-BERT-e2")

model = AutoModel.from_pretrained("KM4STfulltext/SSCI-BERT-e2")
```

- SSCI-SciBERT

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("KM4STfulltext/SSCI-SciBERT-e2")

model = AutoModel.from_pretrained("KM4STfulltext/SSCI-SciBERT-e2")
```

### Download Models

- The version of the model we provide is `PyTorch`. 

### From Huggingface 

- Download directly through Huggingface's official website. 
- [KM4STfulltext/SSCI-BERT-e2](https://huggingface.co/KM4STfulltext/SSCI-BERT-e2)
-  [KM4STfulltext/SSCI-SciBERT-e2](https://huggingface.co/KM4STfulltext/SSCI-SciBERT-e2)
- [KM4STfulltext/SSCI-BERT-e4 ](https://huggingface.co/KM4STfulltext/SSCI-BERT-e4)
- [KM4STfulltext/SSCI-SciBERT-e4](https://huggingface.co/KM4STfulltext/SSCI-SciBERT-e4)

### From Google Drive

We have put the model on Google Drive for users. 

| Model                                                        | DATASET(year) | Base Model             |
| ------------------------------------------------------------ | ------------- | ---------------------- |
| [SSCI-BERT-e2](https://drive.google.com/drive/folders/1xEDnovlwGO2JxqCaf3rdjS2cB6DOxhj4?usp=sharing) | 1986-2021     | Bert-base-cased        |
| [SSCI-SciBERT-e2](https://drive.google.com/drive/folders/16DtIvnHvbrR_92MwgthRRsULW6An9te1?usp=sharing) (recommended) | 1986-2021     | Scibert-scivocab-cased |
| [SSCI-BERT-e4](https://drive.google.com/drive/folders/1sr6Av8p904Jrjps37g7E8aj4HnAHXSxW?usp=sharing) | 1986-2021     | Bert-base-cased        |
| [SSCI-SciBERT-e4](https://drive.google.com/drive/folders/1ty-b4TIFu8FbilgC4VcI7Bgn_O5MDMVe?usp=sharing) | 1986-2021     | Scibert-scivocab-cased |

##  Evaluation & Results

- We use SSCI-BERT and SSCI-SciBERT to perform Text Classificationon different social science research corpus. The experimental results are as follows. Relevant data sets are available for download in the  **Verification task datasets** folder of this project.

#### JCR Title Classify Dataset

| Model                  | accuracy | macro avg | weighted avg |
| ---------------------- | -------- | --------- | ------------ |
| Bert-base-cased        | 28.43    | 22.06     | 21.86        |
| Scibert-scivocab-cased | 38.48    | 33.89     | 33.92        |
| SSCI-BERT-e2           | 40.43    | 35.37     | 35.33        |
| SSCI-SciBERT-e2        | 41.35    | 37.27     | 37.25        |
| SSCI-BERT-e4           | 40.65    | 35.49     | 35.40        |
| SSCI-SciBERT-e4        | 41.13    | 36.96     | 36.94        |
| Support                | 2300     | 2300      | 2300         |

#### JCR Abstract Classify Dataset

| Model                  | accuracy | macro avg | weighted avg |
| ---------------------- | -------- | --------- | ------------ |
| Bert-base-cased        | 48.59    | 42.8      | 42.82        |
| Scibert-scivocab-cased | 55.59    | 51.4      | 51.81        |
| SSCI-BERT-e2           | 58.05    | 53.31     | 53.73        |
| SSCI-SciBERT-e2        | 59.95    | 56.51     | 57.12        |
| SSCI-BERT-e4           | 59.00    | 54.97     | 55.59        |
| SSCI-SciBERT-e4        | 60.00    | 56.38     | 56.90        |
| Support                | 2200     | 2200      | 2200         |

#### JCR Mixed Titles and Abstracts Dataset

| **Model**              | **accuracy** | **macro  avg** | **weighted  avg** |
| ---------------------- | ------------ | -------------- | ----------------- |
| Bert-base-cased        | 58.24        | 57.27          | 57.25             |
| Scibert-scivocab-cased | 59.58        | 58.65          | 58.68             |
| SSCI-BERT-e2           | 60.89        | 60.24          | 60.30             |
| SSCI-SciBERT-e2        | 60.96        | 60.54          | 60.51             |
| SSCI-BERT-e4           | 61.00        | 60.48          | 60.43             |
| SSCI-SciBERT-e4        | 61.24        | 60.71          | 60.75             |
| Support                | 4500         | 4500           | 4500              |

#### SSCI Abstract Structural Function Recognition (Classify Dataset)

|              | Bert-base-cased            | SSCI-BERT-e2        | SSCI-BERT-e4        | support     |
| ------------ | -------------------------- | ------------------- | ------------------- | ----------- |
| B            | 63.77                      | 64.29               | 64.63               | 224         |
| P            | 53.66                      | 57.14               | 57.99               | 95          |
| M            | 87.63                      | 88.43               | 89.06               | 323         |
| R            | 86.81                      | 88.28               | **88.47**           | 419         |
| C            | 78.32                      | 79.82               | 78.95               | 316         |
| accuracy     | 79.59                      | 80.9                | 80.97               | 1377        |
| macro avg    | 74.04                      | 75.59               | 75.82               | 1377        |
| weighted avg | 79.02                      | 80.32               | 80.44               | 1377        |
|              | **Scibert-scivocab-cased** | **SSCI-SciBERT-e2** | **SSCI-SciBERT-e4** | **support** |
| B            | 69.98                      | **70.95**           | **70.95**           | 224         |
| P            | 58.89                      | **60.12**           | 58.96               | 95          |
| M            | 89.37                      | **90.12**           | 88.11               | 323         |
| R            | 87.66                      | 88.07               | 87.44               | 419         |
| C            | 80.7                       | 82.61               | **82.94**           | 316         |
| accuracy     | 81.63                      | **82.72**           | 82.06               | 1377        |
| macro avg    | 77.32                      | **78.37**           | 77.68               | 1377        |
| weighted avg | 81.6                       | **82.58**           | 81.92               | 1377        |

## Cited

- If our content is helpful for your research work, please quote our research in your article. 
- If you want to quote our research, you can use this url ([SsciBERT: A Pre-trained Language Model for Social Science Texts
](https://arxiv.org/abs/2206.04510)) as an alternative before our paper is published.

## Disclaimer

- The experimental results presented in the report only show the performance under a specific data set and hyperparameter combination, and cannot represent the essence of each model. The experimental results may change due to random number seeds and computing equipment. 
- **Users can use the model arbitrarily within the scope of the license, but we are not responsible for the direct or indirect losses caused by using the content of the project.** 


##  Acknowledgment

- SSCI-BERT was trained based on [BERT-Base-Cased]([google-research/bert: TensorFlow code and pre-trained models for BERT (github.com)](https://github.com/google-research/bert)).
- SSCI-SciBERT was trained based on [scibert-scivocab-cased]([allenai/scibert: A BERT model for scientific text. (github.com)](https://github.com/allenai/scibert))

