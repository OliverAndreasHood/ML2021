"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw6"""

#################################
print("\nZadania 1-8b załączono na skanach.\n\nWykonywanie obliczeń na korpisue movie_reviews do następnych zadań.")

import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
import matplotlib.pyplot as plt

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
all_words.most_common(5)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}    
    for w in word_features:
        features[w] = (w in words)
    return features

#ex_r = movie_reviews.words('neg/cv000_29416.txt')
#print(find_features(ex_r)['plot']) 
#print(find_features(ex_r)['prosaic'])

featuresets = [(find_features(rev),category) for (rev,category) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Dokładność metody Naive Bayes do problemu klasfyikacji na zbiorze testowym wynosi:", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)


#################################
print("\nZadanie9.:")
#Zadanie9: Czy fakt, że nie oczyściliśmy danych ze stopwords czy znaków 
#          interpunkcyjncyh znacząco wpłynął na pogorszenie predykcji? 
#          Odpowiedź uzasadnij (bez obliczeń, chodzi o 2 zdania komentarza).

print("Nieoczyszczenie ze stopwords czy interpunkcji nie powinien mieć wpływu na jakosc predykcji.\
 Jak widzimy na ostatnim wypisie, stopwords ani interpunkcja nie była zbyt informatywna\
 o tym czy recenzja była pozytywna czy negatywna. Największe znaczenie miały słowa występujące\
 tylko w jednym z tych typów recenzji, a powyższy wypis pokazuje stosunek występowania\
 danych słów znaczących.")

#################################
print("\nZadanie10.:")
#Zadanie10: Powtórz 20 razy predykcję za pomocą metody Naive Bayes 
#           za każdym razem tasując listę documents (po co?). 
#           Zapisz do listy dokładności uzyskane za każdym razem. 
#           Wyznacz średnią dokładność i odchylenie standardowe z tych d
#           okładności (wcześniej przerób listę na obiekt typu array). 
#           Narysuj histogram dokładności.

def NBsC(i):
    l = []
    for x in range(i):
        random.shuffle(documents)
        featuresets = [(find_features(rev),category) for (rev,category) in documents]
        training_set = featuresets[:1900]
        testing_set = featuresets[1900:]
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        l.append(nltk.classify.accuracy(classifier,testing_set))
        print("Repeats done:",x+1)
    return l

#Długo się to liczy więc skopiowałem jedno przejscie tej funkcji dla i = 20.
NaiveBayVal = [0.87, 0.83, 0.82, 0.82, 0.81, 0.79, 0.82, 0.84, 0.8, 0.85, 0.82, 0.85, 0.76, 0.87, 0.8, 0.78, 0.8, 0.73, 0.83, 0.85]
#NaiveBayVal = NBsC(20)

NaiveBayVal = np.array(NaiveBayVal)
print("Średnia dokładnosć: ",np.mean(NaiveBayVal))
print("Odchylenie standardowe dokładnosci: ",np.std(NaiveBayVal))

plt.hist(NaiveBayVal)
plt.show()


#################################
print("\nZadanie11.: Tworzenie modelu 900:100 [Pos:Neg]")
#Zadanie11: Wyjaśnij czemu istotnym jest aby grupy tekstów reprezentujących 
#           klasy były w miarę równoliczne. Zbuduj model NaiveBayes, 
#           gdzie jako zbiór testowy wybierz 900 recenzji pozytywnych 
#           i 100 negatywnych. Następnie zabadaj jak model sprawdza się 
#           na 100 pozostałych recenzjach pozytywnych a jak na 100 negatywnych
#           (innych niż przy trenowaniu modelu). Skomentuj otrzymane wyniki.

pos_tab = [x for x in featuresets if x[1] == 'pos']
neg_tab = [x for x in featuresets if x[1] == 'neg']

training_set = pos_tab[:900] + neg_tab[:100]
testing_set = pos_tab[900:] + neg_tab[900:]
classifier2 = nltk.NaiveBayesClassifier.train(training_set)
print("Dokładność metody Naive Bayes do problemu klasfyikacji na zbiorze 900:100 [Pos:Neg] wynosi:", (nltk.classify.accuracy(classifier2,testing_set))*100)

print("\nGrupy reprezentujące klasy powinny być w miarę równoliczne, ponieważ na podstawie\
 ich liczebnosci tworzony jest klasyfikator. Większa ilosć testów pozytywnych\
 będzie powodowała zwiększoną dokładnosć ich identyfikacji z jednoczesnym wzrostem\
 prób identyfikowanych jako FalsePositive.\nWynik w wykonanej próbie jest znacznie niższy\
 co jest spowodowane wspomnianym efektem błędnie zbalansowanego klasyfikatora")


#################################
print("\nZadanie12.:")
#Zadanie12: Zbuduj model do predykcji wiadomości SPAM/HAM. W tym celu 
#           użyj pliku  spam_ham.txt . Każda wiadomość opatrzona jest 
#           odpowiednim tagiem na początku kolejnego wiersza. 
#           Sprawdź ile w pliku znajduje się wiadomości typu SPAM i HAM. 
#           Uwzględnij uwagę z poprzedniego zadania.

from nltk import word_tokenize

ham = []
spam = []

with open("spam_ham.txt", "r", encoding='utf8') as f:
    for line in f:
        line = word_tokenize(line.rstrip().lower())
        if 'ham' in line:
            ham.append((line[1:], line[0]))
        elif 'spam' in line:
            spam.append((line[1:], line[0]))

print("\nIlosć SPAM\'u:", len(spam))
print("Ilosć HAM\'u:", len(ham))
msg = ham + spam

#SPAM: 743
#HAM: 4831

hs_all_words = []

for i in range(len(msg)):
    for j in range(len(msg[i][0])):
        hs_all_words.append(msg[i][0][j])

hs_all_words = nltk.FreqDist(hs_all_words)
hs_word_features = [x[0] for x in hs_all_words.most_common(300)]


hs_featuresets = [(find_features(rev),category) for (rev,category) in msg]
training_set = featuresets[:500] + featuresets[750:1250]
testing_set = featuresets[500:700] + featuresets[1250:1450]

classifier3 = nltk.NaiveBayesClassifier.train(training_set)

print("\nDokładność metody Naive Bayes do problemu klasfyikacji wiadomosci SPAM/HAM wynosi:", (nltk.classify.accuracy(classifier3,testing_set))*100)


#################################
print("\nZadanie13.: Zapisywanie modelu.")
#Zadanie13: Zapisz do pliku utworzony poprzednio model (znajdujący się 
#           w pod zmienną  classifier ).

import pickle

modF = open("SH_mod.picle", "wb")
pickle.dump(classifier3, modF)    
modF.close()

print("\nModel zapisano do pliku SH_mod.picle")




























