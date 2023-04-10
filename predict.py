def detect(text):
  import io
  import numpy as np
  import pandas as pd
  import torch
  from transformers import BertTokenizer
  from transformers import BertForSequenceClassification
  from transformers import logging
  logging.set_verbosity_error()
  model1 = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
  model2 = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=3,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
  model1 = model1.to(device)
  model2 = model2.to(device)
  model1.load_state_dict(torch.load('/content/drive/MyDrive/Data/Multi_task_1.model'))
  model2.load_state_dict(torch.load('/content/drive/MyDrive/Data/Multi_task_2.model'))
  tokens = tokenizer.encode(text, return_tensors='pt')
  tokens = tokens.to(device)
  result1 = model1(tokens)
  result2 = model2(tokens)
  suicide = int(torch.argmax(result1.logits))
  sent = int(torch.argmax(result2.logits))
  print('\n')
  if suicide == 0:
    if sent ==0:
      print('Text is Suicidal and has a Negative Sentiment')
    elif sent ==1:
      print('Text is Suicidal and has a Positive Sentiment')
    else:
      print('Text is Suicidal and has a Neutral Sentiment')
  else:
    if sent ==0:
      print('Text is Non-Suicidal and has a Negative Sentiment')
    elif sent ==1:
      print('Text is Non-Suicidal and has a Positive Sentiment')
    else:
      print('Text is Non-Suicidal and has a Neutral Sentiment')


text = input('Enter text:')
detect(text)
