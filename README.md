# Attentive LSTM for question/answering

This is an implementation of the paper - [Improved Representation Learning for Question Answer Matching](http://www.aclweb.org/anthology/P16-1044).
It is implemented on Tensorflow (1.3.0).

## Run Model
> python train.py

## Model Architecture

The model uses bidirectional LSTMs to construct question vector and applies attention on question embedding to contruct answer vector. The loss fuction is the cosine similarity between the question and the answer. For more info, check out the above mentioned paper.

![model](https://github.com/divishdayal/Attentive_LSTM_question-answering/blob/master/model_pic.png "Model Architecture from paper")

## Files -
1. WikiQA-test.tsv - Data for the test dataset
2. WikiQA-train.tsv	- Data for the training dataset
3. config.yml	- Model configurations file for hyperparameter tuning
4. input_wikiqa.py - The text processing file for the dataset used - WikiQA
5. train.py - the file to be executed for training the model
6. model.py - The file containing the model		

