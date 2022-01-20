"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw4"""

#################################
print("\nZadanie1.:")
#Zadanie1: Otworzyć funkcję T/F sprawdzającą czy ciąg jest liczbą czy nie
import string 

def isdig(s):
    if type(s) == int or type(s) == float:
#        print(s,' True')
        return True
    done = []
    for elem in s:
        if elem in string.digits or elem == "," or elem == ".":
            done.append(elem)
        else:
#            print(s,' False')
            return False
#    print(s,' True')
    return True    

#Sprawdzenie podanych przykładów
imp = ['1',1,'12.3',13.3,'50,25','1a']
for i in imp:
    print(i, isdig(i))
    
#################################
print("\nZadanie2.:")
#Zadanie2: wywietl wszystkie konteksty słowa death
from nltk.tokenize import word_tokenize 
from nltk.text import Text
import matplotlib.pyplot as plt

file = open('74457530.txt','r')  
text = file.read().replace('\n','')
file.close()
txt_words = word_tokenize(text)
txt_words = [word.lower() for word in txt_words]   

print(txt_words.count("death"))

t = Text(txt_words)

print(t.concordance('death'))

targets=['woman','disease', 'help', 'bad', 'good','death','date','where']

def dispplt(t_plt, tar_plt):
    from nltk.draw.dispersion import dispersion_plot
    
    plt.figure(figsize=(18, 3))
    dispersion_plot(t_plt, tar_plt, ignore_case=True, title='Lexical Dispersion Plot')
    plt.show()

#dispplt(t, targets)

#################################
print("\nZadanie3.:")
#Zadanie3: wykres częstosci występowania słów bez stop_words, numbers, punctuations
from nltk.corpus import stopwords
sw =  set(stopwords.words("english"))
from nltk.probability import FreqDist


filtered_words = [elem for elem in txt_words if elem not in sw and elem not in string.punctuation and elem not in string.digits and not isdig(elem)]
fdist = FreqDist(filtered_words)
fdist.plot(30,title='Rozkład występowania 30 najpopularniejszych tokenów')


#################################
print("\nZadanie4.:")
#Zadanie4: Kolokacje
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(filtered_words, 5)
finder.apply_freq_filter(10)
print(finder.nbest(bigram_measures.likelihood_ratio, 15))

#################################
print("\nZadanie5.:")
#Zadanie5: Korpus inaugural
import nltk
from nltk.corpus import inaugural
from nltk.tag import pos_tag
from nltk.probability import ConditionalFreqDist

# 1. i 2.
inaugural_tokens = [word.lower() for fileid in inaugural.fileids() for word in inaugural.words(fileid)]

# 3.
print("Ilosc unikalnych tokenów")
print(len(set(inaugural_tokens)))

# 4.
filtered_inaugural_tokens = [elem for elem in inaugural_tokens if elem not in sw and elem not in string.punctuation and elem not in string.digits and not isdig(elem)]
finder = BigramCollocationFinder.from_words(filtered_inaugural_tokens, 5)
finder.apply_freq_filter(10)
print("\n7 Najczęsciej występujących kolokacji")
print(finder.nbest(bigram_measures.likelihood_ratio, 7))

# 5.
alltagged = nltk.pos_tag(inaugural_tokens)
tagged_inaugural_tokens = set(alltagged)

# 6.
quest = ["NN", "NNs"]
nouns = [elem[0] for elem in alltagged if elem[1] in quest]

#nouns_tagged = list(filter(lambda x: x[0] if x[1] in quest, alltagged))
#nouns_n_taggs = list(zip([lambda x: x[0], alltagged],[lambda x: x[1], alltagged]))
#nouns
#
#t1 = list(zip([1,2,3],[4,5,6]))
#t2 = list(zip(*t1))
#
#t2[0]
#t1[0]


# 7.
fdist = FreqDist(nouns)
print('\nTrzy najczęsciej wystepujace rzeczowniki')
print(fdist.most_common(3))

# 8.
targets = [fdist.most_common(3)[i][0] for i in range(3)]
plt.rcParams["figure.figsize"] = (12, 5)
plt.title("Rokład trzech najczęsciej występujących rzeczowników na przestrzeni lat")
cfd = nltk.ConditionalFreqDist((target, fileid[:4])
    for fileid in inaugural.fileids()
    for word in inaugural.words(fileid)
    for target in targets
    if word.lower().startswith(target))

#print('\nRokład tych rzeczowników na przestrzeni lat:')
cfd.plot()

# 9.
from wordcloud import WordCloud

def wcplt(freqdist):
    cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(freqdist)
    plt.figure(figsize=(16,12)) #wymiar obrazka
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
wcplt(fdist)

# 10.
quest = ["JJR", "JJS"]        
jj_set = [elem[0] for elem in tagged_inaugural_tokens if elem[1] in quest]
jj = [elem for elem in inaugural_tokens if elem in jj_set]

fdist_jj = FreqDist(jj)
print(fdist_jj.most_common(3))

# 11.
t2 = Text(inaugural_tokens)
targ2 = ['war']
dispplt(t2,targ2)
