"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw9"""

#################################
print("\nZadanie 1.:")
# Zadanie1: Wygeneruj średnią dokładność dla liczby epok: 
# 1,2,5,10,50,100,1000. 
# Skomentuj otrzymane wyniki.
import numpy as np
import torch
import torch.nn as nn
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

def koszt(ne):
    learning_rate = 0.01
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #Trenowanie
    for epoch in range(ne):
        #Forward
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)

        #Backward
        loss.backward()
        optimizer.step()

        #Zeriwanie grdientu przed kolejnym przejsciem
        optimizer.zero_grad()

        if epoch+1 == ne:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        
    with torch.no_grad():
        y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
        y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
        print(f'accuracy: {acc.item():.4f}')
        
epy = [1,2,5,10,50,100,1000]
for k in epy:
    koszt(k)

print("\nZwiększające się acc, wyjasnia sigmoidalny model funkcji.\n\
Z każdym powtórzeniem wartoci acc zbliżają się do wartosci 1.\n\
Równoczesnie maleje wartosc funkcji kosztu \"loss\".")

#################################
print("\nZadanie 2.:")
# Zadanie2: Zbuduj analogiczny model 
#(oparty o regresje logistyczną i sieci neuronowe) 
# dla recenzji filmowych.
from nltk.corpus import movie_reviews
import nltk

documents = []

print("\nTworzenie documents")
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

print("Tworzenie all_words")
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = []
    for w in word_features:
        if w in words:
            features.append(1)
        else:
            features.append(0)
    return features

print("Tworzenie featuresets_array")
featuresets = [find_features(rev) for (rev,category) in documents]
feat_arr = np.array(featuresets)


pnl = []
# dla nieprzewymieszanych możnaby po prostu macierz 1000x"1" i 1000x"0"
for i in range(len(documents)):
    if documents[i][1] == 'pos':
        pnl.append(1)
    else:
        pnl.append(0)
pn_arr = np.array(pnl)


### Reszta to przekopiowany model z zajęć/poprzedniego zadania.
print("Przygotowanie danych i wytrenowanie modelu\n")
#zmiana X i y na A i b
A, b = feat_arr, pn_arr

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=1234)

#normalizacja danych
sc = StandardScaler()
A_train = sc.fit_transform(A_train)
A_test = sc.transform(A_test)

#zmiana typu - na obiekty typu torch
A_train = torch.from_numpy(A_train.astype(np.float32))
A_test = torch.from_numpy(A_test.astype(np.float32))
b_train = torch.from_numpy(b_train.astype(np.float32))
b_test = torch.from_numpy(b_test.astype(np.float32))

#zmiana wymiaru wektora kategorii
b_train = b_train.view(b_train.shape[0], 1)
b_test = b_test.view(b_test.shape[0], 1)

#Postać modelu wykorzystano z poprzedniego zadania
model = Model(3000)

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy 
optimizer1 = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Trenowanie
for epoch in range(num_epochs):
    #Forward
    y_pred = model(A_train)
    loss = loss_function(y_pred, b_train)

    #Backward
    loss.backward()
    optimizer1.step()

    #Zeriwanie grdientu przed kolejnym przejsciem
    optimizer1.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    b_predicted = model(A_test) #tutaj są liczby z przedzialu (0,1)
    b_predicted_cls = b_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc1 = b_predicted_cls.eq(b_test).sum() / float(b_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc1.item():.4f}')


#################################
print("\nZadanie 3.:")
# Zbuduj analogiczny model 
# (oparty o regresje logistyczną i sieci neuronowe) 
# dla wiadomości SPAM/HAM.

from nltk import word_tokenize

ham = []
spam = []

print("\nŁadowanie i tokenizacja spam_ham.txt")
with open("spam_ham.txt", "r", encoding='utf8') as f:
    for line in f:
        line = word_tokenize(line.rstrip().lower())
        if 'ham' in line:
            ham.append((line[1:], line[0]))
        elif 'spam' in line:
            spam.append((line[1:], line[0]))

#print("\nIlosć SPAM\'u:", len(spam))
#print("Ilosć HAM\'u:", len(ham))
msg = ham + spam

print("Tworzenie hs_all_words i hs_features_array")
hs_all_words = []

for i in range(len(msg)):
    for j in range(len(msg[i][0])):
        hs_all_words.append(msg[i][0][j])

hs_all_words = nltk.FreqDist(hs_all_words)
hs_word_features = [x[0] for x in hs_all_words.most_common(300)]

def find2_features(document):
    words = set(document)
    features = []
    for w in hs_word_features:
        if w in words:
            features.append(1)
        else:
            features.append(0)
    return features

hs_featuresets = [find2_features(rev) for (rev,category) in msg]
feat2_arr = np.array(hs_featuresets)

print("Tworzenie hs_arr")
hsl = []
for i in range(len(msg)):
    if msg[i][1] == 'ham':
        hsl.append(1)
    else:
        hsl.append(0)

hs_arr = np.array(hsl)

### Reszta to przekopiowany model z zajęć/poprzedniego zadania.
print("Przygotowanie danych i wytrenowanie modelu\n")
#zmiana X i y na A i b
C, d = feat2_arr, hs_arr

C_train, C_test, d_train, d_test = train_test_split(C, d, test_size=0.2, random_state=1234)

#normalizacja danych
sc = StandardScaler()
C_train = sc.fit_transform(C_train)
C_test = sc.transform(C_test)

#zmiana typu - na obiekty typu torch
C_train = torch.from_numpy(C_train.astype(np.float32))
C_test = torch.from_numpy(C_test.astype(np.float32))
d_train = torch.from_numpy(d_train.astype(np.float32))
d_test = torch.from_numpy(d_test.astype(np.float32))

#zmiana wymiaru wektora kategorii
d_train = d_train.view(d_train.shape[0], 1)
d_test = d_test.view(d_test.shape[0], 1)

#Postać modelu wykorzystano z poprzedniego zadania
model = Model(300)

#Parametry + funkcja kosztu + optymalizator
num_epochs = 100
learning_rate = 0.01
loss_function = nn.BCELoss() #binary cross-entropy 
optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Trenowanie
for epoch in range(num_epochs):
    #Forward
    y_pred = model(C_train)
    loss = loss_function(y_pred, d_train)

    #Backward
    loss.backward()
    optimizer2.step()

    #Zeriwanie grdientu przed kolejnym przejsciem
    optimizer2.zero_grad()

    if (epoch+1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    d_predicted = model(C_test) #tutaj są liczby z przedzialu (0,1)
    d_predicted_cls = d_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
    acc1 = d_predicted_cls.eq(d_test).sum() / float(d_test.shape[0]) #ile sie zgadza
    print(f'accuracy: {acc1.item():.4f}')

#################################
print("\nZadanie 4.:")
# Zbuduj sieć neuronową opartą o reprezentację BOW, 
# która będzie przewidywała czy fragment tekstu pochodzi z 
# ksiązki Harry Potter czy Władca Pierścieni. 
# Sam zdecyduj co uznasz za jeden rekord. 
# Warto zastosować metody preprocessingu (z biblioteki nltk). 
# Możesz zmieniać zestawy dla współczynnika uczenia, liczby epok, 
# optymalizatora, funkcje kosztu czy zadać parametry początkowe. 
# Efektem powinien być model, który na zbiorze testowym osiąga 
# dokładność co najmniej 70%. Zapisz uzyskany model.

from nltk.tokenize import sent_tokenize
import torch.nn.functional as F
import torch.autograd as autograd
import random
from sklearn.metrics import accuracy_score

#from nltk.stem import PorterStemmer
#import string

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

#Model
class BoWClassifier(nn.Module):  
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim = 1)

def readf(path, tag, out_list):
    if type(path) == str:
        with open(path, "r", encoding = "utf-8") as f:
            whole_text = f.read().replace('\n','').lower()
            sentences = [(word_tokenize(x), tag) for x in sent_tokenize(whole_text)]
                
            for i in sentences:
                out_list.append(i)
    else:
        print("Błędna scieżka do pliku lub zły format")


print("Odczytywanie plików")
lotr_train = []
readf("lotr.txt", 'LOTR', lotr_train)

lotr_test = []
readf("lotr_test.txt", 'LOTR', lotr_test)

hp_train = []
readf("hp.txt", 'HP', hp_train)

hp_test = []
readf("hp_test.txt", 'HP', hp_test)

print("Tworzenie zestawu treningowego i testowego")
# Zestaw treningowy
data_train = lotr_train + hp_train

# Zestaw testowy
data_test = lotr_test + hp_test

random.shuffle(data_test)
random.shuffle(data_train)

# Etykiety
labels = {"LOTR": 0, "HP": 1}

word_to_ix = {}
for sent, _ in data_train + data_test:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
VOCAB_SIZE = len(word_to_ix) #3056 słdów
NUM_LABELS = len(labels) #2 kategorie

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

loss_function = nn.NLLLoss()
optimizer3 = torch.optim.SGD(model.parameters(), lr=0.001)

n_iters = 100

print("Trenowanie..")
for epoch in range(n_iters):
    for instance, label in data_train:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, labels))
        
        #forward
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        
        #backward
        loss.backward()
        optimizer3.step()
        
        #zerujemy gradient
        optimizer3.zero_grad()
        
    if (epoch+1) % (n_iters/10) == 0:
        print(f'> ukonczono: {epoch+1}%')
#list(model.parameters())
print("Uzyskano następujące parametry: \n > ",list(model.parameters())[0].flatten(),"\n >",list(model.parameters())[0][1].flatten(),"\n > ",list(model.parameters())[1].flatten())

E = []
for instance, label in data_test:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    val = [element.item() for element in log_probs.flatten()]
    if val.index(max(val)) == 1:
        E.append(1)
    else:
        E.append(0)

f = []
for i in range(len(data_test)):
    if data_test[i][1] == 'HP':
        f.append(1)
    else:
        f.append(0)

acc = accuracy_score(f, E)
print(f'\nCelnosć: {acc*100:.4f}%')

#accuracy: 78.5679%
#accuracy: 82.1429%
#accuracy: 83.0357%
#accuracy: 81.2500%

try:
    m = "moj_model83.mdl"
    torch.save(model, m)
    print(f"Model zapisano jako: {m}")
except:
    print("Nie udało się zapisać modelu!")
