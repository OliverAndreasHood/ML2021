from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import random
#Ładuje dane
""""
bc = datasets.load_breast_cancer()
print("Features: ", bc.feature_names) #jakie cechy były brane pod uwagę, łącznie 30

print("Labels: ", bc.target_names) #na jakie kategorie dane są podzielone

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

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#ZADANIE 1
#Trenowanie
def difrent_epoch(num_epochs = 100):
    print ("number of epochs: ", num_epochs)
    for epoch in range(num_epochs):
        #Forward
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)

        #Backward
        loss.backward()
        optimizer.step()

        #Zerowanie grdientu przed kolejnym przejsciem
        optimizer.zero_grad()

        #if (epoch+1) % 20 == 0:
            #print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
        y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
        print(f'accuracy: {acc.item():.4f}')

difrent_epoch(1)
difrent_epoch(2)
difrent_epoch(5)
difrent_epoch(10)
difrent_epoch(50)
difrent_epoch(100)
difrent_epoch(1000)
"""
"""
number of epochs:  1
accuracy: 0.3509
number of epochs:  2
accuracy: 0.4123
number of epochs:  5
accuracy: 0.5877
number of epochs:  10
accuracy: 0.7544
number of epochs:  50
accuracy: 0.8596
number of epochs:  100
accuracy: 0.9035
number of epochs:  1000
accuracy: 0.9298
Wraz ze wzrostem liczby epok wzrasta dokładnośc modelu 
"""

#Zadanie 2
'''
from nltk.corpus import movie_reviews
import nltk

#chciałam spróbować na dwa sposoby załadować dane bo to bardzo wolno działa i chyba oba działają tak samo wolno 
#prubowałam dla mniejszej ilosći recenzji i w koncu się udalo, załaczę screena do wiadomośći
print("Features: positive, negative")

document = [(movie_reviews.words(file_id),category) for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id)]
print (document)

most_common_words = []
for word in nltk.FreqDist(movie_reviews.words()).most_common(1000):
    most_common_words.append(word[0])
most_common_words = tuple(most_common_words)

r = []
y = []
t = []
for doc in document:
    if doc[1] == "pos":
        y.append(1)
    else:
        y.append(0)
    t.append(doc[0])

flatten = [item for sublist in t for item in sublist]
most_common_words = []
for word in nltk.FreqDist(flatten).most_common(1000):
    most_common_words.append(word[0])
most_common_words = tuple(most_common_words)

r = []
for tekst in t:
    m = []
    for word in most_common_words:
        if word in tekst:
            m.append(1)
        else:
            m.append(0)
    r.append(m)

X = torch.FloatTensor(r)
y = torch.FloatTensor(y)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
'''
#tutaj ponieżej wersja X i y która działa ale bardzo wolno
'''
for file in positive_review_file:
    t = []
    y.append(1)
    for word in most_common_words:
        if word in movie_reviews.words(f"{file}"):
            t.append(1)
        else:
            t.append(0)
    r.append(t)
for file in negative_review_file:
    t = []
    y.append(0)
    for word in most_common_words:
        if word in movie_reviews.words(f"{file}"):
            t.append(1)
        else:
            t.append(0)
    r.append(t)

X = torch.FloatTensor(r)
y = torch.FloatTensor(y)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

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

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    #Forward
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)

    #Backward
    loss.backward()
    optimizer.step()

    #Zerowanie grdientu przed kolejnym przejsciem
    optimizer.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
    y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc.item():.4f}')

r'''
r'''
#ZADANIE 3
from nltk.corpus import movie_reviews
import nltk
from nltk.tokenize import word_tokenize

print("Features: spam, ham")

with open(r"C:\Users\joann\Downloads\spam_ham.txt", "r", encoding = "utf8") as o:
    t = []
    y = []
    for line in o.readlines():
        words = word_tokenize(line)
        if  words[0] == 'ham':
            y.append(0)
        else:
            y.append(1)
        t.append(words[1:])
print (t)
print (y)
flatten = [item for sublist in t for item in sublist]
most_common_words = []
for word in nltk.FreqDist(flatten).most_common(1000):
    most_common_words.append(word[0])
most_common_words = tuple(most_common_words)
print (most_common_words)

r = []
for tekst in t:
    m = []
    for word in most_common_words:
        if word in tekst:
            m.append(1)
        else:
            m.append(0)
    r.append(m)


X = torch.FloatTensor(r)
y = torch.FloatTensor(y)

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

print (X_train)
print (X_test)
print (y_test)
print (y_train)

#Postać modelu

class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1) #tu: y=wx+b

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x)) #potem licze 1/(1+e^(-y))
        return y_pred

model = Model(n_features)

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    #Forward
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)

    #Backward
    loss.backward()
    optimizer.step()

    #Zerowanie grdientu przed kolejnym przejsciem
    optimizer.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
    y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc.item():.4f}')


'''
#ZADANIE 4
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string



with open(r"C:\Users\joann\Downloads\lor.txt", "r", encoding="utf-8") as f:
    lor = f.read()
with open(r"C:\Users\joann\Downloads\Harry_Potter.txt", "r", encoding="utf-8") as f:
    hp = f.read()

def text(x):
    stop_words = set(stopwords.words("english"))
    not_in = str(stop_words) + string.punctuation
    text = x.replace('\n','')
    txt_words = word_tokenize(x)
    txt_words = [word.lower() for word in txt_words if word not in not_in]
    return txt_words

lor = text(lor)
hp = text(hp)

def make_data(x, y):
    data = []
    for i in range(10, len(x), 10):
        record = x[i-10:i]
        m = (record, y)
        data.append(m)
    return data

data = make_data(hp, "Harry Potter") + make_data(lor, "Lord of Rings")
random.shuffle(data)

test_data = data[:100]
train_data = data[100:]

label_to_ix = { "Harry Potter": 0, "Lord of rings": 1 }

word_to_ix = {}  # tutaj wrzucamy wszystkie slowa, kazde ma indywidualny numer (kolejna liczba naturalna)
for sent, _ in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
vocab_size = len(word_to_ix)
num_labels = len(label_to_ix)
print (vocab_size, num_labels)

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])  #https://pytorch.org/docs/stable/tensors.html


import torch.nn.functional as F #tu są rozne funkcje aktywacji


class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)

model = BoWClassifier(num_labels, vocab_size)
print ("This")
print(list(model.parameters()))

import torch.autograd as autograd
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_iters = 100
for epoch in range(n_iters):
    for instance, label in train_data:
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
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)
print ("elo") # to  tylko chciałam wiedzieć w którym miejscu jestem w wynikach
list(model.parameters())
print (next(model.parameters())[:,word_to_ix["Yes"]])

