# NLP2020-ZAR-pytorch
One challenge for Japanese zero anaphora resolution (ZAR) is the lack of training data, 
making data augmentation a promising approach.   
This repository contains code for the data augmentation by BERT for Japanese ZAR.

## Model Architecture
- BERT as a feature
- k-layer bi-directional GRU + softmax layer

## Data Augmentation
- replacement: Replace token with predicted token by BERT
- zero-drop: Replace embedding with a zero vector
- masking: Replace token with mask-token of masked language model
- masking + zero-drop: combine 'masking' and 'zero-drop'

## Training
The training code is as follows:  
`src/jp_pas/train_pseudo.py`