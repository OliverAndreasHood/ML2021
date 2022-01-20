# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#zadanie 1

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

n_samples, n_features = X.shape #ile probek (569), ile cech (30)

#dziele dane na dwie części: testową i do trenowania, rozmiar zbioru testowego  = 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#normalizacja danych
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#zmiana typu - na obiekty typu torch
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#zmiana wymiaru wektora kategorii
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#Postać modelu

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1) #tu: y=wx+b

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x)) #potem licze 1/(1+e^(-y))
        return y_pred

model = Model(n_features)

def loss_Accuracy(num_epochs):
    #Parametry + funkcja kosztu + optymalizator
    learning_rate = 0.01
    loss_function = nn.BCELoss() #binary cross-entropy 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #Trenowanie

    for epoch in range(num_epochs):
        #Forward
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        #Backward
        loss.backward()
        optimizer.step()

        #Zeriwanie grdientu przed kolejnym przejsciem
        optimizer.zero_grad()
        
        if (epoch+1) == num_epochs:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
        y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
        print(f'accuracy: {acc.item():.4f}')
    
loss_Accuracy(1) #przyklad 1 - epoch =1
loss_Accuracy(2) #przyklad 2 - epoch = 2
loss_Accuracy(5) #przyklad 3 - epoch = 5
loss_Accuracy(10) #przyklad 4 - epoch = 10   
loss_Accuracy(50) #przyklad 5 - epoch = 50
loss_Accuracy(100) #przyklad 6 - epoch = 100
loss_Accuracy(1000) #przyklad 7 - epoch = 1000

#mozna zauwazyc ze dokladnosc sie zwieksza sie wraz ze zwiekszajaca sie iloscia epoch (co mozna bylo na poczatku przewidywac)

#zadanie 2

import nltk
from nltk.corpus import movie_reviews 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

doc_p_n = [] 

for category in movie_reviews.categories(): 
    for fileid in movie_reviews.fileids(category):
        doc_p_n.append((list(movie_reviews.words(fileid)), category)) 

all_words = [] #wszystkie slowa z wszystkich recenzji

for w in movie_reviews.words():  
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
    
word_features = [x[0] for x in all_words.most_common(3000)]

def find_features(document): 
    words = set(document)    
    features = {}            
    for w in word_features:
        if w in words:
            features[w] = 1 
        else:
            features[w] = 0
    return list(features.values())

features_num = [find_features(rev) for (rev,category) in doc_p_n]
features_arr = np.array(features_num)

pos_neg = []
for i in range(len(doc_p_n)):
    if doc_p_n[i][1] == 'pos':
        pos_neg.append(1)
    else:
        pos_neg.append(0)

pos_neg_arr = np.array(pos_neg)

x1, y1 = features_arr, pos_neg_arr

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=1234)

#normalizacja danych
sc1 = StandardScaler()
x1_train = sc1.fit_transform(x1_train)
x1_test = sc1.transform(x1_test)

#zmiana typu - na obiekty typu torch
x1_train = torch.from_numpy(x1_train.astype(np.float32))
x1_test = torch.from_numpy(x1_test.astype(np.float32))
y1_train = torch.from_numpy(y1_train.astype(np.float32))
y1_test = torch.from_numpy(y1_test.astype(np.float32))

#zmiana wymiaru wektora kategorii
y1_train = y1_train.view(y1_train.shape[0], 1)
y1_test = y1_test.view(y1_test.shape[0], 1)

#Postać modelu

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1) #tu: y=wx+b

    def forward(self, x):
        y_pred1 = torch.sigmoid(self.linear(x)) #potem licze 1/(1+e^(-y))
        return y_pred1

model = Model(3000)

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy 
optimizer1 = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Trenowanie
for epoch in range(num_epochs):
    #Forward
    y1_pred = model(x1_train)
    loss1 = loss_function(y1_pred, y1_train)

    #Backward
    loss1.backward()
    optimizer1.step()

    #Zeriwanie grdientu przed kolejnym przejsciem
    optimizer1.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss1.item():.4f}')

with torch.no_grad():
    y1_predicted = model(x1_test) #tutaj są liczby z przedzialu (0,1)
    y1_predicted_cls = y1_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc1 = y1_predicted_cls.eq(y1_test).sum() / float(y1_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc1.item():.4f}')

#zadanie 3

import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#otwieranie pliku i zapisywanie

D_ham = []
D_spam = []

with open("spam_ham.txt", 'r', encoding = "utf-8") as f:
    for line in f:
        if line.startswith('spam'):
            line = word_tokenize(line.rstrip())
            D_spam.append((line[1:], line[0]))
        elif line.startswith('ham'):
            line = word_tokenize(line.rstrip())
            D_ham.append((line[1:], line[0]))

D = D_spam + D_ham

#tworze liste wszystkich slow
ham_spam = []

for i in range(len(D)):
    for h in range(len(D[i][0])):
        ham_spam.append(D[i][0][h])

all_words_hs = [] #wszystkie slowa występujące we wszystkich wiadomosciach

for i in ham_spam: #dla wszystkich slow ze wszystkich wiadomosci
    all_words_hs.append(i.lower())

#czestosc wystepowania wszystkich slow
all_words_hs = nltk.FreqDist(all_words_hs)

#do ponizszej zmiennej zapisuje 300 najczestszych slow
word_features_hs = [x[0] for x in all_words_hs.most_common(300)]

def find_features_hs(document): 
    words = set(document)    
    features = {}            
    for w in word_features_hs:
        if w in words:
            features[w] = 1 
        else:
            features[w] = 0
    return list(features.values())

features_num_hs = [find_features_hs(rev) for (rev,category) in D]

features_arr_hs = np.array(features_num_hs)

h_s = []
for i in range(len(D)):
    if D[i][1] == 'ham':
        h_s.append(1)
    else:
        h_s.append(0)

h_s_arr = np.array(h_s)

x2, y2 = features_arr_hs, h_s_arr

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=1234)

#normalizacja danych
sc2 = StandardScaler()
x2_train = sc2.fit_transform(x2_train)
x2_test = sc2.transform(x2_test)

#zmiana typu - na obiekty typu torch
x2_train = torch.from_numpy(x2_train.astype(np.float32))
x2_test = torch.from_numpy(x2_test.astype(np.float32))
y2_train = torch.from_numpy(y2_train.astype(np.float32))
y2_test = torch.from_numpy(y2_test.astype(np.float32))

#zmiana wymiaru wektora kategorii
y2_train = y2_train.view(y2_train.shape[0], 1)
y2_test = y2_test.view(y2_test.shape[0], 1)

#Postać modelu

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1) #tu: y=wx+b

    def forward(self, x):
        y_pred1 = torch.sigmoid(self.linear(x)) #potem licze 1/(1+e^(-y))
        return y_pred1

model = Model(300)

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy 
optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Trenowanie
for epoch in range(num_epochs):
    #Forward
    y2_pred = model(x2_train)
    loss2 = loss_function(y2_pred, y2_train)

    #Backward
    loss2.backward()
    optimizer2.step()

    #Zeriwanie grdientu przed kolejnym przejsciem
    optimizer2.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss2.item():.4f}')

with torch.no_grad():
    y2_predicted = model(x2_test) #tutaj są liczby z przedzialu (0,1)
    y2_predicted_cls = y2_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc2 = y2_predicted_cls.eq(y2_test).sum() / float(y2_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc2.item():.4f}')
  
#zadanie 4

import random
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


lotr_all_tr = []
with open("lotr.txt", 'r', encoding = "utf-8") as f1:
    for line in f1:
        if not line.isspace():
            line = word_tokenize(line.rstrip())
            line2 = []
            for word in line:
                if word.isalpha() == True:
                    line2.append(word)
            lotr_all_tr.append((line2, 'LOTR'))

lotr_all_test = []
with open("lotr_test.txt", 'r', encoding = "utf-8") as f2:
    for line in f2:
        if not line.isspace():
            line = word_tokenize(line.rstrip())
            line2 = []
            for word in line:
                if word.isalpha() == True:
                    line2.append(word)
            lotr_all_test.append((line2, 'LOTR'))

hp_all_tr = []
with open("hp.txt", 'r', encoding = "utf-8") as f3:
    for line in f3:
        if not line.isspace():
            line = word_tokenize(line.rstrip())
            line2 = []
            for word in line:
                if word.isalpha() == True:
                    line2.append(word)
            hp_all_tr.append((line2, 'HP'))
            
hp_all_test = []
with open("hp_test.txt", 'r', encoding = "utf-8") as f4:
    for line in f4:
        if not line.isspace():
            line = word_tokenize(line.rstrip())
            line2 = []
            for word in line:
                if word.isalpha() == True:
                    line2.append(word)
            hp_all_test.append((line2, 'HP'))
            
train_list = lotr_all_tr + hp_all_tr
test_list = lotr_all_test + hp_all_test

#tasuje liste do treningu i testowa
random.shuffle(train_list)
random.shuffle(test_list)

label_to_ix = { "LOTR": 0, "HP": 1}

word_to_ix = {} #tutaj wrzucamy wszystkie slowa, kazde ma indywidualny numer (kolejna liczba naturalna)
for sent, _ in train_list+test_list:
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

import torch.nn.functional as F #tu są rozne funkcje aktywacji

#Model

class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim = 1)
    
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

import torch.autograd as autograd

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_iters = 100
for epoch in range(n_iters):
    for instance, label in train_list:     
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
  
#zapisuje tu wszystkie log_probs w postaci tensorow
all_log_probs = []

for instance, label in test_list:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    all_log_probs.append(log_probs)

#pozbywam sie tensorow i wyciagam z nich odpowiednie wartosci
val = []    
for elem in all_log_probs:
    lista = [element.item() for element in elem.flatten()]
    val.append(lista)

#dopasowuje odpowiedni numer w zaleznosci od tego ktory element danej sublisty byl wiekszy
pred_cat = []
for i in range(len(val)):
    a = val[i].index(max(val[i]))
    if a == 1:
        pred_cat.append(1)
    else:
        pred_cat.append(0)

#tworze liste kategorii (ich numerow) na podstawie test_list     
real_categories = []
for i in range(len(test_list)):
    if test_list[i][1] == 'HP':
        real_categories.append(1)
    else:
        real_categories.append(0)
        
#licze dokladnosc metody
accuracy_score(real_categories, pred_cat) ##wyszla mi dokladnosc okolo 78%

#zapisuje model i sprawdzam, czy dziala
torch.save(model, 'pc_model')
m2  = torch.load('pc_model')
list(m2.parameters())
