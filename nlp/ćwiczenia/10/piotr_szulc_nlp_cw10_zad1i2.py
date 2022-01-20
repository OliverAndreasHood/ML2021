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
        print("out embendingi: ", embeds)
        out = F.relu(self.linear1(embeds))  #dzialam funkcją aktywacji ReLu na wynik po pierwszym przeksztalceniu liniowym
        print("out relu: ", out)
        out = self.linear2(out)             #przeksztalcam liniowo poprzednie wartosci
        print("out linear2: ", out)
        log_probs = F.log_softmax(out, dim=1) #na koniec wyznaczam logarytmy prawdopodobienstw
        print("log_probs: ", log_probs)
        return log_probs

#Dane do modelu

CS = 2 #ile slow uzywam do predykcji
ED = 10 #rozmiar osadzonego wektora (embedding vector)
HD = 256 #rozmiar warstwy ukrytej

learning_rate = 0.001
loss_function = nn.NLLLoss()  #funkcja kosztu

model = NGramModel(len(vocab), ED, CS) 
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#print(model.eval()) 

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
print(list(model.parameters()))
#print(list(model.parameters())[0][5])
#print(list(model.parameters())[1][5])
#print(list(model.parameters())[2][5])
#print(list(model.parameters())[3][5])
#print(list(model.parameters())[4][5])

#print("CS: ", CS, "\tED: ", ED, "\tHD: ", HD)
#print(len(list(model.parameters())[0]), "x", len(list(model.parameters())[0][0]))
#print(len(list(model.parameters())[1]), "x", len(list(model.parameters())[1][0]))
#print(len(list(model.parameters())[2]))
#print(len(list(model.parameters())[3]), "x", len(list(model.parameters())[3][0]))
#print(len(list(model.parameters())[4]))

try:
    m = "moj_model0.mdl"
    torch.save(model, m)
    print(f"Model zapisano jako: {m}")
except:
    print("Nie udało się zapisać modelu!")


#print("\nZadanie: Wyjaśnij powyższe wyniki")#
# CS:  2         ED:  4         HD:  3
# 58 x 4
# 3 x 8
# 3
# 58 x 3
# 58

##Model
#class NGramModel(nn.Module):
#
#    def __init__(self, vocab_size, embedding_dim, context_size):
#        super().__init__()
#        self.embeddings = nn.Embedding(vocab_size, embedding_dim) #slowa na embeddingi
#        self.linear1 = nn.Linear(context_size * embedding_dim, HD) #pierwsze przeksztalcenie liniowe
#        self.linear2 = nn.Linear(HD, vocab_size)  #drugie przeksztalcenie liniowe
#
#    def forward(self, inputs):
#        embeds = self.embeddings(inputs).view((1, -1)) #embeddingi
#        out = F.relu(self.linear1(embeds))  #dzialam funkcją aktywacji ReLu na wynik po pierwszym przeksztalceniu liniowym
#        out = self.linear2(out)             #przeksztalcam liniowo poprzednie wartosci
#        log_probs = F.log_softmax(out, dim=1) #na koniec wyznaczam logarytmy prawdopodobienstw
#        return log_probs
#
#
##> self.embeddings = nn.Embedding(vocab_size, embedding_dim)
##Embeddingi: vocab_size, CS -> odpowiada tensorowi pierwszemu z parametrów bo dla każdego unikalnego słowa przypisujemy wektor 10-ciu parametrów (tensor 58x10)
#
##> self.linear1 = nn.Linear(context_size * embedding_dim, HD)
##Linear1: CS * ED, HD -> odpowiada drugiemu tensorowi z parametrów bo bierzemy pod uwagę kontekst (2 słowa) przerzucając do warstwy ukrytej jako trzy zmienne funkcji liniowej (3 wartosci).
##Uzyskujemy tensor warstwy ukrytej o wymiarach: HD = 3, context(2) * embedding(4) czyli 2x4=8 => tensor 3x8
#
##> out = F.relu(self.linear1(embeds))
## działamy funkcją ReLu po pierwszym przekształceniu liniowym (funkcja aktywacji => max(0,y))

embedd = torch.tensor([[ 0.2481,  0.3692, -0.7330,  0.4940, -0.8680, 
                        0.7258, -0.5098, -1.2559, 0.8786, -0.4926,  
                        0.0186,  0.6130,  1.4379, -0.0066, -0.7198,  
                        0.4909, -0.9187, -1.0668,  0.4317,  1.1668]])
w1 = model.linear1.weight
b1 = model.linear1.bias

















#print("> W wyjasnieniu pomoże nam wypisanie długosci uzyskanych tensowór odzwierciedlających wyuczone parametry:\n")
#print("Długosć pierwszego tensora:", len(list(model.parameters())[0]), "\n> Odpowiada embeddingom po pierwszej linearyzacji (58 wektorów 10-cio wymiarowych)\n")
#print("Długosć drugiego tensora:", len(list(model.parameters())[1]), "\n> Odpowiada warstwie ukrytej w której przeprowadzane są wyszystkie przeliczenia, wynik działania funkcji RELU (256 wektorów 20-sto wymiarowych)\n")
#print("Długosć trzeciego tensora:", len(list(model.parameters())[2]), "\n> Tensor wartosci wagowych poszczególnych embeddingów\n")
#print("Długosć czwartego tensora:", len(list(model.parameters())[3]), "\n> Efekt drugiej linearyzacji - podsumowanie wyników przeliczeń z warstwy ukrytej (58 wektorów 256-cio wymiarowych\n")
#print("Długosć piątego tensora:", len(list(model.parameters())[4]), "\n> Softmax layer -> logarytmy wyliczonych prawdopodobieństw. Na tej podstawie uzyskany model przewiduje kolejne słowo tekstu.\n")
#
#
###################################
#print("\nZadanie 2.:")
## Zdefiniuj model, który będzie przewidywał słowo \
## w oparciu o 2 wcześniejsze i 2 następne słowa.
#
#vocab = list(set(test_sentence))
#word_to_ix = {word: i for i, word in enumerate(vocab)}
#N5_GM = list(ngrams(test_sentence,5))
#N5_GM = [([a,b,d,e],c) for a,b,c,d,e in N5_GM]
#
##Dane do modelu
#CS = 4 #ile slow uzywam do predykcji
#ED = 10 #rozmiar osadzonego wektora (embedding vector)
#HD = 256 #rozmiar warstwy ukrytej
#
#learning_rate = 0.001
#loss_function = nn.NLLLoss()  #funkcja kosztu
#
#model2 = NGramModel(len(vocab), ED, CS) 
#optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate)
#
##Postać modelu
#print("Postać przygotowanego modelu:\n", model2.eval()) 
#
#for epoch in range(100):
#    for context, target in N5_GM:
#        
#        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)  
#        log_probs = model2(context_idxs)                                                  
#        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long)) 
#        
#        #liczenie pochodnych i aktualizacja parametrow
#        loss.backward()
#        optimizer2.step()  
#        
#        #zerowanie gradientu
#        optimizer2.zero_grad()
#
#
#with torch.no_grad():
#    context2 = ['then', 'is', 'no', 'more'] #"heard"
#    #context2 = ['out', 'brief', "life's", 'but'] #'candle', (możliwosc sprawdzenia czy działa na innym zestawie)
#    
#    context2_idxs = torch.tensor([word_to_ix[w] for w in context2], dtype=torch.long) 
#    print(context2_idxs) 
#    pred2 = model2(context2_idxs) 
#    print(pred)                
#    index_of_prediction2 = numpy.argmax(pred2)
#    print(vocab[index_of_prediction2])
#    print("> Przewidywane słowo: ", context2[0], context2[1], "\"", vocab[index_of_prediction2], "\"", context2[2], context2[3])






















