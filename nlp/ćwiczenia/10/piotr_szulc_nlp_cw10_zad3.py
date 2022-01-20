#################################
print("\nZadanie 3.:")
# Zbuduj model przewidujący czy dana recenzja jest pozytywna \
# czy negatywna w oparciu o reprezentacje wektorową słów (embeddingi) \
# wytrenowanych na Wikipedia 2014 oraz sieć neuronową. \
# Wygeneruj wykresy wartości funkcji kosztu vs liczby epok \
# oraz Dokładności vs liczby epok. Dokładność na zbiorze testowym \
# powinna wynosić co najmniej 70 %.

import torchtext
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random

#import recenzji do reviews = []
from nltk.corpus import movie_reviews

reviews = []

print("\nTworzenie reviews")
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        reviews.append((list(movie_reviews.words(fileid)), category))


# postać reviews (niefiltrowane):
# [([lista słów i znaków recenzji 1], pos/neg), ([lista słów i znaków recenzji 2], pos/neg), ...]

print("Tworzenie filtrowanych reviews")
GF_ref = []

for revid in range(len(reviews)):
    GF_ref.append(([elem for elem in reviews[revid][0] if elem.isalpha()], reviews[revid][1]))

# postać GF_ref (filtrowane reviews):
# [([lista tylko słów recenzji 1], pos/neg), ([lista tylko słów recenzji 2], pos/neg), ...]

random.shuffle(GF_ref)

print("Import GloVe dim=50")
#glove -> zawiera wszystkie wytrenowane embeddingi wielkosci dim=50!
glove = torchtext.vocab.GloVe(name="6B", dim=50)

def get_reviews_vectors(glove_vector):
    train, valid, test = [], [], [] #tworze trzy listy na dane train, valid i test
    
    for i, line in enumerate(GF_ref): #przechodze dane 
        rev = line[0]                 #tekst recenzji to ostatni element 
        rev_emb = sum(glove_vector[w] for w in rev) #wektor recenzji będzie sumą embeddingow slow w niego wchodzących
#        label = line[1] #generuje dwa rodzaje labelow: 1 - gdy happy, 0 gdy sad
        label = torch.tensor(int(line[1] == "pos")).long()
        #dzielimy dane na trzy kategorie
        if i % 5 < 3:     #czyli 0,1,2                   
            train.append((rev_emb, label)) # 60% danych treningowych
        elif i % 5 == 3: #czyli 3
            valid.append((rev_emb, label)) # 20% danych do walidacji
        else:            #czyli gddy i%5 jest 4
            test.append((rev_emb, label)) # 20% danych testowych
    return train, valid, test


print("Przygotowanie danych: train, valid, test")
#przygotowuje sobie dane w oparciu o gotowe embeddingi z glove
train, valid, test = get_reviews_vectors(glove)

#len(train) #1200
#len(valid) #400
#len(test)  #400

print("Tworzenie train_loader, valid_loader, test_loader")
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

#trenowanie
def train_network(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-5):
    print("\nTrenowanie: przygotowanie zmiennych")
    criterion = nn.CrossEntropyLoss()  #funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optymalizator ADAM
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    
    print("Trenowanie modelu:")
    for epoch in range(num_epochs):          #dla kazdej epoki
        for revs, labels in train_loader:  #przechodze dane treningowe
            optimizer.zero_grad()
            pred = model(revs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            
        losses.append(float(loss))           #zapisuje wartosc funkcji kosztu
        if epoch % 20 == 0:                  #co 20 epoke 
            epochs.append(epoch)             #zapisz numer epoki
            train_acc.append(get_accuracy(model, train_loader))   #dokladnosc na zbiorze testowym
            valid_acc.append(get_accuracy(model, valid_loader))   #dokladnosc na zbiorze treningowym
            print(f'Epoch number: {epoch+1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1],3)} | Valid accuracy: {round(valid_acc[-1],3)}')
    
    #Rysowanie wykresow
    plt.title("Training Curve")
    plt.plot(losses, label="Train dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train dataset")
    plt.plot(epochs, valid_acc, label="Validation dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

#Funkcja wyznaczająca dokładność predykcji:
def get_accuracy(model, data_loader):
    correct, total = 0, 0  #ile ok, ile wszystkich
    for revs, labels in data_loader: #przechodzi dane
        output = model(revs)         #jak dziala model
        pred = output.max(1, keepdim=True)[1]  #ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total



a = []
for i in range(10):
    z = random.randrange(10,50)
    a.append(([z], "model"+str(i+1)))


acc_s = [[],[],[]]    
for z,m in a:
    print("Parametr wyjsciowy: ",z[0])
    val = int(z[0])
    #Siec neuronową moge zapisac w taki uproszczony spoosob
    mymodel0 = nn.Sequential(nn.Linear(50, val),
                            nn.ReLU(),
                            nn.Linear(val, 2))
    
    mymodel1 = nn.Sequential(nn.Linear(50, val),
                            nn.ReLU(),
                            nn.Linear(val, int(val/2)),
                            nn.Linear(int(val/2), 2))
    
    mymodel2 = nn.Sequential(nn.Linear(50, val),
                            nn.ReLU(),
                            nn.Linear(val, int(val/2)),
                            nn.Linear(int(val/2), int(val/4)),
                            nn.Linear(int(val/4), 2))

    train_network(mymodel0, train_loader, valid_loader, num_epochs=100, learning_rate=0.001)
    acc = get_accuracy(mymodel0, test_loader)
    print("Final test accuracy:", acc)
    acc_s[0].append([[val], acc])
        
    try:
        torch.save(mymodel0, m+"_0.mdl")
        print(f"Model zapisano jako: {m}_0.mdl")
    except:
        print("Nie udało się zapisać modelu!")
    
    train_network(mymodel1, train_loader, valid_loader, num_epochs=200, learning_rate=0.001)
    acc = get_accuracy(mymodel1, test_loader)
    print("Final test accuracy:", acc)
    acc_s[1].append([[val, int(val/2)], acc])
        
    try:
        torch.save(mymodel0, m+"_1.mdl")
        print(f"Model zapisano jako: {m}_1.mdl")
    except:
        print("Nie udało się zapisać modelu!")

    train_network(mymodel2, train_loader, valid_loader, num_epochs=200, learning_rate=0.001)
    acc = get_accuracy(mymodel2, test_loader)
    print("Final test accuracy:", acc)
    acc_s[2].append([[val, int(val/2), int(val/4)], acc])
        
    try:
        torch.save(mymodel0, m+"_2.mdl")
        print(f"Model zapisano jako: {m}_2.mdl")
    except:
        print("Nie udało się zapisać modelu!")








#    mymodel = nn.Sequential(nn.Linear(50, 40),
#                            nn.ReLU(),
#                            nn.Linear(40, 30),
#                            nn.ReLU(),
#                            nn.Linear(30, 20),
#                            nn.ReLU(),
#                            nn.Linear(20, 10),
#                            nn.ReLU(),
#                            nn.Linear(10, 2))
# 70,25%

#    mymodel = nn.Sequential(nn.Linear(50, 24),
#                            nn.ReLU(),
#                            nn.Linear(24, 2))
# 70,25%



