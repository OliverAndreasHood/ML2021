# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:18:30 2020

@author: amand
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import torch.autograd as autograd
import torch.nn.functional as F 
import random
###################### HP TRAINING ##############################

with open("hp.txt", 'r', encoding = "utf-8") as f1:
    text = f1.read()
    

hp_sent_tokenize = sent_tokenize(text)

tokens_h = [w.lower() for w in hp_sent_tokenize]
table = str.maketrans('', '', string.punctuation)
tokens_h = [w.translate(table) for w in tokens_h]
tokens_h = [word for word in tokens_h if not word.isalpha()]

hp_clear = []
for i in tokens_h:
    n = i.replace("\n", " ")
    hp_clear.append(n)

hp_c = [word_tokenize(i) for i in hp_clear]   
hp_training = []
for elem in hp_c:
    hp_training.append((elem, "HP"))


############# HP TEST #########################

with open("hp_test.txt", 'r', encoding = "utf-8") as f2:
    text1 = f2.read()
    

hp_test_sent_tokenize = sent_tokenize(text1)

tokens_h_test = [w.lower() for w in hp_test_sent_tokenize]
table = str.maketrans('', '', string.punctuation)
tokens_h_test = [w.translate(table) for w in tokens_h_test]
tokens_h_test = [word for word in tokens_h_test if not word.isalpha()]

hp_test_clear = []
for i in tokens_h_test:
    n1 = i.replace("\n", " ")
    hp_test_clear.append(n1)

hp_test_c = [word_tokenize(i) for i in hp_test_clear]   
hp_test = []
for elem in hp_test_c:
    hp_test.append((elem, "HP"))


############# LOTR TRAINING ##########################  
with open("lotr.txt", 'r', encoding = "utf-8") as f3:
    text2 = f3.read()
    

lotr_sent_tokenize = sent_tokenize(text2)

tokens_l = [w.lower() for w in lotr_sent_tokenize]
table = str.maketrans('', '', string.punctuation)
tokens_l = [w.translate(table) for w in tokens_l]
tokens_l = [word for word in tokens_l if not word.isalpha()]

lotr_clear = []
for i in tokens_l:
    n2 = i.replace("\n", " ")
    lotr_clear.append(n2)

lotr_c = [word_tokenize(i) for i in lotr_clear]   
lotr_training = []
for elem in hp_c:
    lotr_training.append((elem, "LOTR"))


############## LOTR TEST ########################    
with open("lotr_test.txt", 'r', encoding = "utf-8") as f4:
    text3 = f4.read()
    

lotr_test_sent_tokenize = sent_tokenize(text3)

tokens_l_test = [w.lower() for w in lotr_test_sent_tokenize]
table = str.maketrans('', '', string.punctuation)
tokens_l_test = [w.translate(table) for w in tokens_l_test]
tokens_l_test = [word for word in tokens_l_test if not word.isalpha()]

l_test_clear = []
for i in tokens_l_test:
    n3 = i.replace("\n", " ")
    l_test_clear.append(n3)

l_test_c = [word_tokenize(i) for i in l_test_clear]   
l_test = []
for elem in l_test_c:
    l_test.append((elem, "LOTR"))


#####################################

training = hp_training + lotr_training
testing = hp_test + l_test
random.shuffle(training)
random.shuffle(testing)
label_to_ix = { "LOTR": 0, "HP": 1}

word_to_ix = {} #tutaj wrzucamy wszystkie slowa, kazde ma indywidualny numer (kolejna liczba naturalna)
for sent, _ in training+testing:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
VOCAB_SIZE = len(word_to_ix)  #ile wszysktich slow
NUM_LABELS = len(label_to_ix) #2 - ile kategorii

#pomocnicza funkcja1

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

#pomocnicza funkcja2

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])  #https://pytorch.org/docs/stable/tensors.html



#Model

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim = 1)
    
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_iters = 200
for epoch in range(n_iters):
    for instance, label in training:     
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
    
        #forward
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        
        #backward
        loss.backward()
        optimizer.step()
        
        #zerujemy gradient
        optimizer.zero_grad()
        
for instance, label in testing:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)

#wszystkie log_probs
calc_log_probs = []

for instance, label in testing:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    calc_log_probs.append(log_probs)
    
#bez tensorów
calc_probs = []    
for i in calc_log_probs:
    lista = [y.item() for y in i.flatten()]
    calc_probs.append(lista)

#teoretyczne dopasowanie
pred_cat = []
for i in range(len(calc_probs)):
    a = calc_probs[i].index(max(calc_probs[i]))
    if a == 1:
        pred_cat.append(1)
    else:
        pred_cat.append(0)

#tworze liste kategorii (ich numerow) na podstawie test_list     
real_cat = []
for i in range(len(testing)):
    if testing[i][1] == 'HP':
        real_cat.append(1)
    else:
        real_cat.append(0)
        
#dokladosc
accuracy_score(real_cat, pred_cat) 
print(accuracy_score(real_cat, pred_cat))


