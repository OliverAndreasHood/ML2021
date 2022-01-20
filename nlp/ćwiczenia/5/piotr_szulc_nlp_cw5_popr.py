"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw5"""

#################################
print("\nZadanie1.:")
#Zadanie1: treebank
from nltk.corpus import treebank
from nltk.tag import hmm

trained = treebank.tagged_sents()[:3000]
print("trained\tdone")
tested = treebank.tagged_sents()[3000:]
print("tested\tdone")

trainer = hmm.HiddenMarkovModelTrainer()
print("trainer\tdone")
tagger = trainer.train_supervised(trained)
print("tagger\tdone")

x = tagger.evaluate(trained)
print("Trained evaluate: ", x)
x2 = tagger.evaluate(tested)
print("Tested evaluate: ", x2)

print("Ponieważ model nauczony na \"trained\" zna niejako słowa \
      zawarte w tej zmiennej, wynik ewaluacji na zbiorze \"tested\"\
      musi dać niższy wynik.")

#################################
print("\nZadanie2.:")
#Zadanie2:  Chunking
from nltk.chunk import RegexpParser
from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag

text = "The Very Large Telescope (VLT) of the European Southern Observatory (ESO), an array of four individual telescopes in the Atacama desert, has given us a huge amount of new data about the universe. Researchers have now used it to find a group of six galaxies around a supermassive black hole, from when the Universe was just 0.9 billion years old - it's estimated to be 13.8 billion years old now. Black holes are thought to sit at the center of galaxies including the Milky Way. "
sentences = sent_tokenize(text.lower())
tokens_sen1 = word_tokenize(sentences[0])
tags = pos_tag(tokens_sen1)

grammar = "chunk: {<DT>?<JJ>*<NN>}"
chunker = RegexpParser(grammar) 
result1 = chunker.parse(tags)
#result1.draw()

def lschunk(r):
    z2l = [subtree.leaves() for subtree in r.subtrees() if subtree.label() == 'chunk']
    z2l_wynik = []
    for elem in z2l:
        s = ''
        for i in range(len(elem)):
            s = s + ' ' + elem[i][0] 
        z2l_wynik.append(s)
    return z2l_wynik
    
print(lschunk(result1))

#################################
print("\nZadanie3.:")
#Zadanie3: {<NN.?>*<VBD.?>*<JJ.?>*<CC>?}
grammar = "chunk: {<NN.?>*<VBD.?>*<JJ.?>*<CC>?}" 
chunker = RegexpParser(grammar) 
result2 = chunker.parse(tags)
#result2.draw()

print(lschunk(result2))

#import nltk.help
#nltk.help.upenn_tagset('NN')
#nltk.help.upenn_tagset('VBD')
#nltk.help.upenn_tagset('JJ')
#nltk.help.upenn_tagset('CC')

print("{<NN.?>*<VBD.?>*<JJ.?>*<CC>?} oznacza:\n\
      <NN.?>* - tag NN (rzeczownik) . (cokolwiek) ? (jedno lub żadne wystąpienie) * (żadne lub dowolnej długosci powtorzenie) \n\
      <VBD.?>* - tag VBD (czasownik) reszta j.w.\n\
      <JJ.?>* - tag JJ (przymiotnik lub liczebnik) reszta j.w.\n\
      <CC>? - tag CC (spójnik) ? (żadne lub jedno wystąpienie)")

#################################
print("\nZadanie4.:")
#Zadanie4: plik i własny chunk
with open("tekst1.txt", "r") as f:
    text = f.read().replace('\n','')

sentences = sent_tokenize(text.lower())
tokens_sen2 = word_tokenize(sentences[1]) 
tags2 = pos_tag(tokens_sen2)

grammar2 = "chunk: {<DT>?<JJ.?>*<NN.?>*<VBD.?>*<CC>?}" 
chunker = RegexpParser(grammar) 
result3 = chunker.parse(tags2)
#result3.draw()
print(lschunk(result3))

#################################
print("\nZadanie5.:")
#Zadanie5: entity w txt
from nltk import ne_chunk

txt = "Google, LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware. It is considered one of the Big Five technology companies in the U.S. information technology industry, alongside Amazon, Facebook, Apple, and Microsoft."

tokens_sen3 = word_tokenize(txt)
tags3 = pos_tag(tokens_sen3)
namedEnt = ne_chunk(tags3, binary = True)

z5l = []
for i in range(len(namedEnt)):
    try:
        if namedEnt[i].label() == 'NE':
            z5l.append(namedEnt[i][0][0])
    except:
        continue

print(z5l)

#################################
print("\nZadanie6.:")
#Zadanie6: Stemming
from nltk.stem import PorterStemmer

def StemMachine(s):
    ps = PorterStemmer()
    s_tokens = word_tokenize(s)
    r = ''
    i = 0
    for elem in s_tokens:
        x = ps.stem(elem)
        if i == 0:
            r += x
            i += 1
            continue
        if x == ".":
            r += x
        else: 
            r = r + " " + x
    return(r)

txt2 = "Dancing is an art. Students should be taught dance as a subject in schools. I danced in many of my school function. Some people are always hesitating to dance."
print(StemMachine(txt2))

#################################
print("\nZadanie7.:")
#Zadanie7: Lemmatization
from nltk.stem import WordNetLemmatizer

def LemMachine(s):
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(s)
    tab = []
    
    for i in range(len(tokens)):
        if pos_tag(word_tokenize(tokens[i]))[0][1][0] == 'R':
            tab.append(lemmatizer.lemmatize(tokens[i], pos = 'r'))
        elif pos_tag(word_tokenize(tokens[i]))[0][1][0] == 'N':
            tab.append(lemmatizer.lemmatize(tokens[i], pos = 'n'))
        elif pos_tag(word_tokenize(tokens[i]))[0][1][0] == 'V':
            tab.append(lemmatizer.lemmatize(tokens[i], pos = 'v'))
        elif pos_tag(word_tokenize(tokens[i]))[0][1][0] == 'J':
            tab.append(lemmatizer.lemmatize(tokens[i], pos = 'a'))
        else:
            tab.append(tokens[i])
    
    return " ".join(tab)
    
txt3 = "The striped bats are hanging on their feet for best"
print(LemMachine(txt3))

#################################
print("\nZadanie8.:")
#Zadanie8: hiponimy i hiperonimy słowa dog
from nltk.corpus import wordnet as wn

#wn.synsets('dog')
ss = 'dog.n.01'

#wn.synset(ss).lemma_names() 
#wn.synset(ss).definition()
#wn.synset(ss).examples()

pet = wn.synset(ss)
z8l_hpo = [elem.name() for elem in pet.hyponyms()]
z8l_hpr = [elem.name() for elem in pet.hypernyms()]

print("Hyponyms: \n", z8l_hpo)
print("Hypernyms: \n", z8l_hpr)

#################################
print("\nZadanie9.:")
#Zadanie9: synonimi i antonimy

def syms(s):
    synonyms = []
    for syn in wn.synsets(s):
        for lemma in syn.lemmas(): 
            synonyms.append(lemma.name())
    return synonyms

def anms(s):
    antonyms = []
    for syn in wn.synsets(s):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return antonyms

word = 'happy'
print(syms(word))
print(anms(word))

#################################
print("\nZadanie10.:")
#Zadanie10: porównanie dog, cat, fish

dogo = wn.synsets('dog')[0]
kitku = wn.synsets('cat')[0]
bulbu = wn.synsets('fish')[0]

print('dogo do kitku: ', dogo.wup_similarity(kitku))
print('dogo do bulbu: ', dogo.wup_similarity(bulbu))
print('kitku do bulbu: ', kitku.wup_similarity(bulbu))

#################################
print("\nZadanie11.:")
#Zadanie11: NGramy
from nltk import ngrams
from collections import Counter

def LastMachine(text, N):
    NgramT = []
    
    for word in word_tokenize(ex_text):
        NgramT.append(list(ngrams(word, 4, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_')))
    
    NgramT = [word for sublist in NgramT for word in sublist]
    list_NgramT = NgramT
    
    for idx, val in enumerate(NgramT):
        list_NgramT[idx] = ''.join(val)
        
    end = Counter(list_NgramT)
    return dict(end)

ex_text = "This is an example paper in which I summed up some basic concepts reffering to viruses."
print(LastMachine(ex_text, 4))










