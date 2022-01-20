# -*- coding: utf-8 -*-

#zadanie 1

from nltk.corpus import treebank
from nltk.tag import hmm

trained = treebank.tagged_sents()[:3000]
tested = treebank.tagged_sents()[3000:]

#budowanie hmm na trained

trainer = hmm.HiddenMarkovModelTrainer() 
tagger = trainer.train_supervised(trained)

#ewaluacja dla trained - sprawdzanie przewidywalnosci
tagger.evaluate(trained)

#ewaluacja dla tested - sprawdzanie przewidywalnosci
tagger.evaluate(tested)

'''
w przypadku przewidywalnosci istnieje spora rozbieznosc
miedzy przewidywalnoscia dla trained i tested:
dazaca do jedynki przewidywalnosc w przypadku trained
jest spowodowana tym, ze to na otagowanych slowach z trained
'trenowalismy' caly model - zatem jego przewidywalnosc bedzie
duza na liscie, na ktorej trenowal. z kolei, w przypadku
tested - model ma do czynienia z lista, na ktorej nie trenowal
- zatem mozna bylo sie spodziwac, ze przewidywalnosc dla tej listy bedzie
duzo mniejsza niz dla listy, na ktorej model sie uczyl
'''

#zadanie 2

from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser

text = "The Very Large Telescope (VLT) of the European Southern Observatory (ESO), an array of four individual telescopes in the Atacama desert, has given us a huge amount of new data about the universe. Researchers have now used it to find a group of six galaxies around a supermassive black hole, from when the Universe was just 0.9 billion years old - it's estimated to be 13.8 billion years old now. Black holes are thought to sit at the center of galaxies including the Milky Way. "

#wszystkie chunki w tekscie
tokens_sen1 = word_tokenize(text.lower())
tags = pos_tag(tokens_sen1)

#szukanie chunkow spelniajacych ponizsze zalozenie
grammar = "chunk: {<DT>?<JJ>*<NN>}" 
chunker = RegexpParser(grammar) 
result = chunker.parse(tags)

for subtree in result.subtrees():
    if subtree.label() == 'chunk':
        print(subtree.leaves())

#zadanie 3

'''
wyrazenie regularne do interpretacji: 
    <NN.?>*<VBD.?>*<JJ.?>*<CC>?
    
pierwszy <>: NN lub NN + jeden dowolny znak, 
* za nim oznacza żadne lub dowolnej długosci powtorzenie wyrazenia w <>

drugi <>: analogicznie, ale w przypadku VBD

trzeci <>: analogicznie, ale w przypadku JJ

czwarty <>: ? za <> oznacza, że wyrazenie w <>
ma powtorzyc sie raz albo wcale

'''

#zadanie 4

from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.chunk import RegexpParser

file = open('C:\\Users\Paulina\Desktop\Bioinformatyka\otw\\74457530.txt','r')  #plik z cwiczen czwartych
t = file.read().replace('\n','')
file.close()

#wszystkie chunki w danym zakresie tekstu: wybrałam zakres, zeby szybciej liczyl
tokens_sen2 = word_tokenize(t.lower()[:3000])
tags2 = pos_tag(tokens_sen2)

#szukam wybranego chunka
grammar2 = "chunk: {<JJ?>*<DT?>?<NN?>*}" 
chunker2 = RegexpParser(grammar2) 
result2 = chunker2.parse(tags2)

for subtree in result2.subtrees():
    if subtree.label() == 'chunk':
        print(subtree.leaves())

#zadanie 5

from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

txt = "Google, LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five technology companies in the U.S. information technology industry, alongside Amazon, Facebook, Apple, and Microsoft."

#tworzenie tagow i tokenow
tokens_sen3 = word_tokenize(txt)
tags3 = pos_tag(tokens_sen3)

#szukanie bytow (entities)
namedEnt = ne_chunk(tags3, binary = True)

#zadanie 6

from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

napis = "Dancing is an art. Students should be taught dance as a subject in schools. I danced in many of my school function. Some people are always hesitating to dance."

def stemming_zd(n):
    ps = PorterStemmer()
    a = []
    b = word_tokenize(n)
    for elem in b:
        a.append(ps.stem(elem))
    
    g = ' '.join(a)
    g = g.replace(" .", ".")
    return g

stemming_zd(napis)

#zadanie 7

def lematyz_zd(n):
    
    from nltk import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    
    tok_sen = word_tokenize(n)
    lemmatizer = WordNetLemmatizer()
    zlem = []
    
    for i in range(len(tok_sen)):
        
        if pos_tag([tok_sen[i]])[0][1][0] == 'R': #przyslowki
            zlem.append(lemmatizer.lemmatize(tok_sen[i], pos = 'r'))
        elif pos_tag([tok_sen[i]])[0][1][0] == 'V': #czasowniki
            zlem.append(lemmatizer.lemmatize(tok_sen[i], pos = 'v'))
        elif pos_tag([tok_sen[i]])[0][1][0] == 'N': #rzeczowniki
            zlem.append(lemmatizer.lemmatize(tok_sen[i], pos = 'n'))
        elif pos_tag([tok_sen[i]])[0][1][0] == 'J': #przymiotniki
            zlem.append(lemmatizer.lemmatize(tok_sen[i], pos = 'a'))
        else:
            zlem.append(tok_sen[i])

    zlem_sklad = ' '.join(zlem)
    
    return zlem_sklad

sth = "The striped bats are hanging on their feet for best"

lematyz_zd(sth)

#zadanie 8

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

#szukanie synsetow dla slowa 'dog'
dog_synsets = wn.synsets("dog")

dog_syn_name = wn.synset('dog.n.01')

#hiponimy dla slowa dog (z pierwszego synsetu)
hipo_dog = dog_syn_name.hyponyms()

hipon = [lemma.name() for synset in hipo_dog for lemma in synset.lemmas()]

#hiperonimy dla slowa dog (z pierwszego synsetu)
hiper_dog = dog_syn_name.hypernyms()

hiper = [lemma.name() for synset in hiper_dog for lemma in synset.lemmas()]

#zadanie 9

#szukanie synonimow i antonimow

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

synonyms = []
for syn in wn.synsets('happy'):
    for lemma in syn.lemmas(): 
        synonyms.append(lemma.name())

print(synonyms)

antonyms = []
for syn in wn.synsets("happy"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(antonyms)

#zadanie 10

from nltk.corpus import wordnet as wn

dog = wn.synsets('dog')[0]
cat = wn.synsets('cat')[0]
fish = wn.synsets('fish')[0]

#podobienstwo dog, cat i fish
dog.wup_similarity(fish), cat.wup_similarity(fish), cat.wup_similarity(dog)

#zadanie 11

from nltk import ngrams
from nltk import sent_tokenize, word_tokenize
from collections import Counter

ex_text = "This is an example paper in which I summed up some basic concepts reffering to viruses."

def N_grams(text, N):
    
    generated_ngrams = [] 
    
    for word in word_tokenize(text):
        generated_ngrams.append(list(ngrams(word, N, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_')))

    generated_ngrams = [word for sublist in generated_ngrams for word in sublist] 
    
    ng_list_ngrams = generated_ngrams
    
    for idx, val in enumerate(generated_ngrams):
        ng_list_ngrams[idx] = ''.join(val)
    
    counting = Counter(ng_list_ngrams)
    

    return dict(counting)
    
N_grams(ex_text, 4)