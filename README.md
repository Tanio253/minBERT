# minBERT
As the original BERT model has two objective function: masked sentence and next predicting sentence. Here we just implement the first objective function.
First run source setup.py to setup all the packages required.
Here I have set up 10 test cases for the encoder layer sanity check.
For training minbert we will use 2 dataset: Stanford Sentiment Treebank (SST) dataset and CFIMDB dataset
After finetune for the sentiment analysis we then continue finetune for multitask: sentiment analysis with sst dataset, paraphase detection with quora dataset, semantic texual analysis with sts dataset'
