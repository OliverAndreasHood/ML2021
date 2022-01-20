# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:15:55 2020

@author: amand
"""
#1
###################### IMPORT
import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#####################################################

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

def loss(num_epochs):
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

    #Zerwanie gradientu przed kolejnym przejsciem
        optimizer.zero_grad()

        if epoch+1 == num_epochs:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test) #tutaj są liczby z przedzialu (0,1)
        y_predicted_cls = y_predicted.round() #zaokrągla je do 0 lub 1 [kategorii]
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) #ile sie zgadza
        print(f'accuracy: {acc.item():.4f}')
        
loss(1)
loss(2)
loss(5)
loss(10)
loss(50)
loss(100)
loss(1000)
'''
Wraz ze wzrostem liczby epoch, nasza funkcja kosztu zbliża się ku 0,a accuracy sukcesywnie zbliższa się do 1.
'''


