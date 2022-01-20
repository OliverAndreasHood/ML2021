"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw12"""

#################################
print("\nZadanie 1.:")
# prawdopodobieństwo pozostawienia słowa

import numpy as np

def P(w):
    return min((np.sqrt(w/0.001)+1)*0.001/w,1)

print("P(0.04):\t",P(0.04))
print("P(1):\t",P(1))

print("P(x) = 0.5 dla x = 0.0074641:\t",P(0.0074641))
print("P(x) = 1 dla x = 0.0026180:\t",P(0.0026180))

#################################
print("\nZadanie 2.:")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torchtext
print("przypisanie GloVe")
glove = torchtext.vocab.GloVe(name="6B", dim=50)


#Wykluczone:
#'Andorra la Vella', 'Andorra', 
#'Sarajevo', 'Bosnia and Hercegovina'
#'Prague', 'Czech Republic'
#'San Marino', 'San Marino'
#'Vatican City', 'Holy See',

#Dla Państwo-Stolica
#words = ['netherlands', 'amsterdam', 'greece', 'athens', 'serbia', 'belgrade', 'germany', 'berlin', 'switzerland', 'bern', 'slovakia', 'bratislava', 'belgium', 'brussels', 'romania', 'bucharest', 'hungary', 'budapest', 'moldova', 'chisinau', 'denmark', 'copenhagen', 'ireland', 'dublin', 'finland', 'helsinki', 'ukraine', 'kiev', 'portugal', 'lisbon', 'slovenia', 'ljubljana', 'uk', 'london', 'luxembourg', 'luxembourg', 'spain', 'madrid', 'belarus', 'minsk', 'monaco', 'monaco', 'russia', 'moscow', 'cyprus', 'nicosia', 'greenland', 'nuuk', 'norway', 'oslo', 'france', 'paris', 'montenegro', 'podgorica', 'czech', 'prague', 'iceland', 'reykjavik', 'latvia', 'riga', 'italy', 'rome', 'bulgaria', 'sofia', 'sweden', 'stockholm', 'estonia', 'tallinn', 'albania', 'tirana', 'liechtenstein', 'vaduz', 'malta', 'valletta', 'austria', 'vienna', 'lithuania', 'vilnius', 'poland', 'warsaw', 'croatia', 'zagreb']

#co ciekawe wykres dla powyższego zestawu państw-miast ma całkiem inną orientację punktów w przestrzeni (poziomą)


def wykresikPCA(w):
    X = []
    for word in w:
        X.append(glove[word].tolist()) #patrze jakie mają embeddingi
    
    X = np.array(X) #tworze z nich macierz

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)  #redukuje liczbe zmiennych z 50 do 2 dla kazdego slowa


    #Do rysowania: 
	
	#fig = plt.subplots(1,1, figsize=(30,20))
    x = X_pca[:,0] #pierwsza zmienna
    y = X_pca[:,1] #druga zmienna

    plt.scatter(x,y)

    for i, txt in enumerate(w):
        plt.annotate(txt, (x[i], y[i]))
    
    for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')

	
    #plt.savefig('European_Capitols.png', bbox_inches='tight')
    plt.show()


words1 = ['france', 'paris', 'poland', 'warsaw', 'germany', 'berlin', 'netherlands', 'amsterdam']
wykresikPCA(words1)


#################################
print("\nZadanie 3.:")

from sklearn.manifold import MDS

def wykresikMDS(w):
    X = []
    for word in w:
        X.append(glove[word].tolist()) #patrze jakie mają embeddingi
    
    X = np.array(X) #tworze z nich macierz

    X_MDS = MDS(n_components=2).fit_transform(X)  #redukuje liczbe zmiennych z 50 do 2 dla kazdego slowa

    #Do rysowania: 
	#fig = plt.subplots(1,1, figsize=(30,20))
	
    x = X_MDS[:,0] #pierwsza zmienna
    y = X_MDS[:,1] #druga zmienna

    plt.scatter(x,y)

    for i, txt in enumerate(w):
        plt.annotate(txt, (x[i], y[i]))
    
    for i in range(0, len(x), 2):
        plt.plot(x[i:i+2], y[i:i+2], 'ro-')


    #plt.savefig('European_Capitols.png', bbox_inches='tight')
    plt.show()

words2 = ['dog', 'cat', 'cow', 'horse', 'sheep', 'sunflower', 'carrot', 'salad', 'potato','cucumber']
wykresikMDS(words2)
wykresikPCA(words2)

print("Metoda MDS nie jest powtarzalna, za każdym razem zwraca inną macierz embeddingów. \n\
Mamy niepowtarzalne wyniki jednak w większosci przypadków można rozróżnić poszczególne grupy \
czy to zwierząt czy roslin. \n\
Stosując metodę PCA wyniki są powtarzalne i bardzo łatwo rozróżnić poszczególne grupy.")
























