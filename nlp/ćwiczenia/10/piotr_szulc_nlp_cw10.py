"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw10"""

#################################
print("\nZadanie 1.:")
# Na podstawie budowy modelu oraz jego parametrów \
# wyjaśnij uzyskane powyżej wartości \
# (Uwaga: przy każdym generowaniu modelu \
# otrzymasz inny zestaw parametrów).

from nltk import ngrams
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

test_sentence = ['tomorrow', 'and', 'tomorrow', 'and', 'tomorrow', 'creeps', 'in', 'this', 'petty', 'pace', 'from', 'day', 'to', 'day', 'to', 'the', 'last', 'syllable', 'of', 'recorded', 'time', 'and', 'all', 'our', 'yesterdays', 'have', 'lighted', 'fools', 'the', 'way', 'to', 'dusty', 'death', 'out', 'out', 'brief', 'candle', "life's", 'but', 'a', 'walking', 'shadow,', 'a', 'poor', 'player', 'that', 'the','way', 'to','struts', 'and', 'frets', 'his', 'hour', 'upon', 'the', 'stage', 'and', 'then', 'is', 'heard', 'no', 'more', 'it', 'is', 'a', 'tale', 'told', 'by', 'an', 'idiot', 'full', 'of', 'sound', 'and', 'fury', 'signifying', 'nothing']

vocab = list(set(test_sentence))
word_to_ix = {word: i for i, word in enumerate(vocab)}
N3_GM = list(ngrams(test_sentence,3))
N3_GM = [([x,y],z) for x,y,z in N3_GM]

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

#Dane do modelu

CS = 2 #ile slow uzywam do predykcji
ED = 10 #rozmiar osadzonego wektora (embedding vector)
HD = 256 #rozmiar warstwy ukrytej

learning_rate = 0.001
loss_function = nn.NLLLoss()  #funkcja kosztu

model = NGramModel(len(vocab), ED, CS) 
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
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


import numpy

#Przewidywanie wg modelu

with torch.no_grad():
    #bede przewidywal kolejne slowo na podstawe tych dwoch
    context = ['the', 'way']  
    #zapisuje w odpowiednim formacie te dane
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long) 
    #tak wygladają te dane  [ich indeksy]
    print(context_idxs) 
    #co pokazuje model?
    pred = model(context_idxs) 
    #co przewiduje model?
    print(pred)                
    #wyznaczam ten indeks gdzie jest największa wartosc
    index_of_prediction = numpy.argmax(pred)  
    print("> Przewidywane słowo: ", context[0], context[1], "\"", vocab[index_of_prediction], "\"")

#Uzyskane wyniki przewidywaniaprzewidywania:
# tensor([39,  2])
# tensor([[-4.5336, -4.9523, -4.5546, -4.7472, -5.2571, -4.5710, -5.3834, -5.4415, -4.6181, -4.2471, -4.9537, -4.9978, -4.8862, -3.7339, -5.2767, -5.1522, -4.6843, -5.0461, -4.8240, -5.5300, -4.4798, -5.0141, -4.6284, -5.5006, -4.6480, -5.1412, -4.9098, -4.8360, -5.1323, -5.0143, -4.1581, -5.2239, -4.9485, -4.9864, -3.9895, -4.9288, -4.4801, -4.7676, -4.7909, -3.3801, -5.4523, -5.0493, -5.2123, -2.5992, -5.1891, -5.4403, -5.0155, -5.0566, -4.9609, -0.8027, -5.1388, -5.1144, -4.3549, -4.3826, -4.6136, -5.6776, -5.3187, -4.7657]])

print("\nZadabnie: Wyjaśnij powyższe wyniki")

print("> W wyjasnieniu pomoże nam wypisanie długosci uzyskanych tensowór odzwierciedlających wyuczone parametry:\n")
print("Długosć pierwszego tensora:", len(list(model.parameters())[0]), "x", len(list(model.parameters())[0][0]), "\n> Odpowiada embeddingom wszystkich unikatowych slow (58 wektorów 10-cio wymiarowych)\n")
print("Długosć drugiego tensora:", len(list(model.parameters())[1]), "x", len(list(model.parameters())[1][0]), "\n> Tensory po pierwszej linearyzacji (256 wektorów 20-sto wymiarowych)\n")
print("Długosć trzeciego tensora:", len(list(model.parameters())[2]), "\n> Tensor wartosci BIAS z pierwszej linearyzacji (256 wartosci b)\n")
print("Długosć czwartego tensora:", len(list(model.parameters())[3]), "x", len(list(model.parameters())[3][0]), "\n> Wyniki drugiej linearyzacji (58 wektorów 256-cio wymiarowych\n")
print("Długosć piątego tensora:", len(list(model.parameters())[4]), "\n> Tensor wartosci BIAS z drugiej linearyzacji\n")


#################################
print("\nZadanie 2.:")
# Zdefiniuj model, który będzie przewidywał słowo \
# w oparciu o 2 wcześniejsze i 2 następne słowa.

vocab = list(set(test_sentence))
word_to_ix = {word: i for i, word in enumerate(vocab)}
N5_GM = list(ngrams(test_sentence,5))
N5_GM = [([a,b,d,e],c) for a,b,c,d,e in N5_GM]

#Dane do modelu
CS = 4 #ile slow uzywam do predykcji
ED = 10 #rozmiar osadzonego wektora (embedding vector)
HD = 256 #rozmiar warstwy ukrytej

learning_rate = 0.001
loss_function = nn.NLLLoss()  #funkcja kosztu

model2 = NGramModel(len(vocab), ED, CS) 
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate)

#Postać modelu
print("Postać przygotowanego modelu:\n", model2.eval()) 

for epoch in range(100):
    for context, target in N5_GM:
        
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)  
        log_probs = model2(context_idxs)                                                  
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long)) 
        
        #liczenie pochodnych i aktualizacja parametrow
        loss.backward()
        optimizer2.step()  
        
        #zerowanie gradientu
        optimizer2.zero_grad()


with torch.no_grad():
    context2 = ['then', 'is', 'no', 'more'] #"heard"
    #context2 = ['out', 'brief', "life's", 'but'] #'candle', (możliwosc sprawdzenia czy działa na innym zestawie)
    context2_idxs = torch.tensor([word_to_ix[w] for w in context2], dtype=torch.long) 
    print(context2_idxs) 
    pred2 = model2(context2_idxs) 
    print(pred)                
    index_of_prediction2 = numpy.argmax(pred2)
    print(vocab[index_of_prediction2])
    print("> Przewidywane słowo: ", context2[0], context2[1], "\"", vocab[index_of_prediction2], "\"", context2[2], context2[3])


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

print("Import GloVe dim=100")
#glove -> zawiera wszystkie wytrenowane embeddingi wielkosci dim=50!
glove = torchtext.vocab.GloVe(name="6B", dim=100)

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

mymodel = nn.Sequential(nn.Linear(100, 70),
                        nn.ReLU(),
                        nn.Linear(70, 50),
                        nn.ReLU(),
                        nn.Linear(50, 20),
                        nn.ReLU(),
                        nn.Linear(20, 10),
                        nn.ReLU(),
                        nn.Linear(10, 2))
    
train_network(mymodel, train_loader, valid_loader, num_epochs=100, learning_rate=0.001)
acc = get_accuracy(mymodel, test_loader)
print("Final test accuracy:", acc)

try:
    m = "moj_model0.mdl"
    torch.save(mymodel, m)
    print(f"Model zapisano jako: {m}")
except:
    print("Nie udało się zapisać modelu!")













