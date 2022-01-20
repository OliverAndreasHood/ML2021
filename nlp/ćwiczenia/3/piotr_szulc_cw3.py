"""Zadania Piotr Szulc"""

#################################
print("\nZadanie1.:")
#Zadanie1: porównaj sent_tokenize i split('.')

from nltk.tokenize import sent_tokenize, word_tokenize

s = "Mr. Smith is a scientist. He is also a very good teacher."
t1 = s.split('.')
t2 = sent_tokenize(s)

print(t1,"\n",t2)

#################################
print("\nZadanie2.:")
#Zadanie2: funkcja lower

def low_tokenz(txt):
    try:
        from nltk.tokenize import word_tokenize
    except:
        import nltk
        nltk.download('punkt')
    
    txt = txt.lower()
    txt = word_tokenize(txt)
    return txt

print(low_tokenz(s))

#################################
print("\nZadanie3.:")
#Zadanie3: Load 74457530.txt & process

with open("74457530.txt", "r") as f:
    txt_words = low_tokenz(f.read())
    
print(txt_words[:20])
M = len(txt_words)
V = len(set(txt_words))
#Liczba Herdana
import math as mt
C = mt.log(V)/mt.log(M)
print("\nM: ",M,"\nV: ",V,"\nC: ",C)
print("\"woman\"_count", txt_words.count('woman'))

#################################
print("\nZadanie4.:")
#Zadanie4: text without SP
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words("english"))
ex_sentence = "This is an example 1, showing off stop words filtration that occur using NLTK library."
words_withoutSP = []
for elem in word_tokenize(ex_sentence):
    if elem not in stop_words:
        if elem not in string.digits:
            if elem not in string.punctuation:
                words_withoutSP.append(elem)
print(words_withoutSP)

#################################
print("\nZadanie5.:")
#Zadanie5: słownik stopwords'ów
from nltk.corpus import PlaintextCorpusReader
corpus_root = 'nls-text-indiaPapers/'
wordlists = PlaintextCorpusReader(corpus_root, '.*', encoding='latin1')
corpus_tokens = wordlists.words()

lct = [word.lower() for word in corpus_tokens]

swlib1 = {}
for elem in lct:
    if elem in stop_words:
         if elem in swlib1:
             swlib1[elem] += 1
         else:
             swlib1[elem] = 1
             
max_value = max(swlib1.values())
max_keys = [k for k, v in swlib1.items() if v == max_value]
print(max_keys, max_value)

""" 
import datetime   
print(datetime.datetime.now())
print(corpus_tokens.count('the'))
print(datetime.datetime.now())

for elem in stop_words:
    swlib[elem] = corpus_tokens.count(elem)
    #Zdecydowanie za długo (+/- 2h liczenia)
"""

#################################
print("\nZadanie6.:")
#Zadanie6: przykłady wyrazów spełniających wyrażenia
print("smakkkk, oma") #'.ma.*'
print("mea, meal") #'meal?'
print("google, goooooogle") #'go{2,6}gle'
print("ake, ane, ame") #'a[knm]e'
print("bed, bud") #'b[^a]d'

#################################
print("\nZadanie7.:")
#Zadanie7: przeszukiwanie z wrozcem wom[ae]n
import re
womaen_strings=[w for w in txt_words if re.search('^wom[ae]n$', w)]
womaen2_strings=[w for w in txt_words if re.search('wom[ae]n', w)]

print(womaen_strings)
print(womaen2_strings)

#################################
print("\nZadanie8.:")
#Zadanie8: przeszukiwanie ^a.{13}
a13_s = [w for w in txt_words if re.search('^a.{13}', w)]
print(a13_s)

#################################
print("\nZadanie9.:")
#Zadanie9: przeszukiwanie [^a-z]
male13 = [w for w in txt_words if re.search('^[^a-z].{12}', w)]
print(male13)















