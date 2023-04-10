# Multi-task-Suicide-Prediction-BERT
Multi-Task Learning to Train BERT transformers for predicting suicidal intentions from Reddit posts using sentiment analysis as the auxiliary task

Please refer https://www.kaggle.com/datasets/amangoyl/reddit-dataset-for-multi-task-nlp for dataset

## Short description of notebooks:


Sentiment_labels.ipynb -- Labelling the data with sentiments using Transfer Learning and making three models to vote for the final label

Multi-Task_Suicide_Sentiment.ipynb -- Implementation of Multi-task Learning model

Baseline_SVM_Suicide_Sentiment.ipynb -- Experiments with SVM model for both Suicidal detection as well as Sentiment analysis

Suicide_detection_single_task_plus_huggingface.ipynb -- Single task Suicide Detection experiments along with experiments related pre-trained hugging-face transformer model

Sentiment_Prediction_Single_task.ipynb -- Single task Sentiment prediction experiments

## Performance
F1 Score Suicidal Multi-task	0.970


F1 Score Sentiment Multi-task	0.891
