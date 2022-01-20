# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:08:18 2020

@author: pauli
"""

#ZADANIE 1

test_sentence = ['tomorrow', 'and', 'tomorrow', 'and', 'tomorrow', 'creeps', 'in', 'this', 'petty', 'pace', 'from', 'day', 'to', 'day', 'to', 'the', 'last', 'syllable', 'of', 'recorded', 'time', 'and', 'all', 'our', 'yesterdays', 'have', 'lighted', 'fools', 'the', 'way', 'to', 'dusty', 'death', 'out', 'out', 'brief', 'candle', "life's", 'but', 'a', 'walking', 'shadow,', 'a', 'poor', 'player', 'that', 'the','way', 'to','struts', 'and', 'frets', 'his', 'hour', 'upon', 'the', 'stage', 'and', 'then', 'is', 'heard', 'no', 'more', 'it', 'is', 'a', 'tale', 'told', 'by', 'an', 'idiot', 'full', 'of', 'sound', 'and', 'fury', 'signifying', 'nothing']

vocab = list(set(test_sentence)) #lista slow unikalnych

word_to_ix = {word: i for i, word in enumerate(vocab)} #kazdemu slowu przyporządkowuje unikalny numer

from nltk import ngrams
N3_GM = list(ngrams(test_sentence,3)) #3-gramy

N3_GM = [([x,y],z) for x,y,z in N3_GM] #to samo co wyzej, ale nieco w innym w formacie
print(N3_GM[:3])

import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
#Model
class NGramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) #slowa na embeddingi
        self.linear1 = nn.Linear(context_size * embedding_dim, HD) #pierwsze przeksztalcenie liniowe
        self.linear2 = nn.Linear(HD, vocab_size)  #drugie przeksztalcenie liniowe

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1)) #embeddingi
        out = F.relu(self.linear1(embeds))  #dzialam funkcją aktywacji ReLu na wynik po pierwszym przeksztalceniu liniowym
        out = self.linear2(out)             #przeksztalcam liniowo poprzednie wartosci
        log_probs = F.log_softmax(out, dim=1) #na koniec wyznaczam logarytmy prawdopodobienstw
        return log_probs
    
CS = 2
ED = 4
HD = 3

learning_rate = 0.001
loss_function = nn.NLLLoss()  #funkcja kosztu

model = NGramModel(len(vocab), ED, CS) 
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#Postać modelu
print(model.eval()) 


for epoch in range(50):
    for context, target in N3_GM:
        
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)  #dane
        log_probs = model(context_idxs)                                                  #co daje model
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long)) #funkcja kosztu
        
        #liczenie pochodnych i aktualizacja parametrow
        loss.backward()
        optimizer.step()  
        
        #zerowanie gradientu
        optimizer.zero_grad()
#parametry po wytrenowaniu modelu

print(list(model.parameters()))

import numpy

#Przewidywanie wg modelu

with torch.no_grad():
    context = ['the', 'way']  #bede przewidywal kolejne slowo na podstawe tych dwoch
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long) #zapisuje w odpowiednim formacie te dane
    print(context_idxs) #tak wygladają te dane  [ich indeksy]
    pred = model(context_idxs) #co pokazuje model?
    print(pred)                #co przewiduje model?
    index_of_prediction = numpy.argmax(pred)  #wyznaczam ten indeks gdzie jest największa wartosc
    print(vocab[index_of_prediction])         #jakie slowo sie tam kryje
    

#zadanie 1

context = ['the', 'way']  #bede przewidywal kolejne slowo na podstawe tych dwoch
context_idxs = numpy.array([word_to_ix[w] for w in context]) #zapisuje w odpowiednim formacie te dane

#embeddingi z wytrenowanego modelu znajdują sie tutaj
print(list(model.parameters())[0])
em = list(model.parameters())[0]

#w tym wypadku chodzi mi o embeddingi dla indeksow 48 i 34, wiec wyciagam je na potrzeby dalszych obliczen
val = []    
for elem in em:
    lista = [element.item() for element in elem.flatten()]
    val.append(lista)

#UWAGA! za kazdym razem moze zmienić indeks naszych slow, wiec najpierw wywolac print(context_idxs) i sprawdzic, jakie sa tam numerki i wpisac je do emb1 i emb2
emb1 = numpy.array(val[45])
emb2 = numpy.array(val[37])

#wagi biorę z naszego wytrenowanego modelu, tak samo bias


#dla 1 przeksztalcenia liniowego
weights = model.linear1.weight

num_wei = []
for elem in weights:
    lista = [element.item() for element in elem.flatten()]
    num_wei.append(lista)
    
w_L1 = numpy.array(num_wei)   

bias1 = model.linear1.bias

b1 = []
for elem in bias1:
    lista = [element.item() for element in elem.flatten()]
    b1.append(lista)

b_L1 = numpy.array(b1)


#dla 2 przeksztalcenia liniowego
weights2 = model.linear2.weight

num_wei2 = []
for elem in weights2:
    lista = [element.item() for element in elem.flatten()]
    num_wei2.append(lista)

w_L2 = numpy.array(num_wei2)

bias2 = model.linear2.bias

b2 = []
for elem in bias2:
    lista = [element.item() for element in elem.flatten()]
    b2.append(lista)

b_L2 = numpy.array(b2)

S = emb1
S1 = numpy.append(S, emb2)

#szukam argumentow warstwy ukrytej - poprzez 1 przeksztalcenie liniowe
h1 = numpy.dot(w_L1[0], numpy.transpose(S1)) + b_L1[0]
h2 = numpy.dot(w_L1[1], numpy.transpose(S1)) + b_L1[1]
h3 = numpy.dot(w_L1[2], numpy.transpose(S1)) + b_L1[2]

#warstwa ukryta jako jeden wektor
H1 = h1
H2 = numpy.append(H1, h2)
H = numpy.append(H2, h3)

#dzialam funkcja aktywacji ReLU na wynik w H
H = numpy.array([max(elem,0) for elem in H])

#wykonuje drugie przeksztalcenie liniowe
M = []
for i in range(len(b_L2)):
    a = numpy.dot(w_L2[i],numpy.transpose(H))+b_L2[i]
    M.append(a.tolist())

#zamieniam liste na array, zeby na niej liczyc
M = numpy.array(M)

#ostatni etap - przepuszczam wartosci przez softmax

SUMA = sum(numpy.e**x for x in M)

print([numpy.log((numpy.e**x)/SUMA) for x in M])
