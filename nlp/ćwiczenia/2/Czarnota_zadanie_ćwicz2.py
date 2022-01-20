# -*- coding: utf-8 -*-

#zadanie 1

fun = lambda x: (3*(x**2))-2
print(list(fun(elem) for elem in range(1,6)))

#inne rozwiazanie

print(list((3*(elem**2))-2 for elem in range(1,6)))

#zadanie 2

print(list(map(lambda a: a.count('T'), ["ATTGC", "AGGC", "TTTGC"]))) #policz ile w danym elemencie listy jest tymin

#zadanie 3

print(list(filter(lambda x: x%2 == 0, range(1,41)))) #znajdz liczby parzyste

#zadanie 4

#znajdz podsekwencje ATG w sekwencjach i pokaz te, ktore ja zawieraja
print(list(filter(lambda a: a.find("ATG") > -1 , ["ATGCC", "TATC", "GGATGGGG"])))

#zadanie 5

nazwy = ['seq1', 'seq2']
sekw = ['ATGGGC', 'AGGGCT']

f = open("seq.fa", "w")
for i in range(0,2):
    f.write(">" + nazwy[i] + "\n" + sekw[i] + "\n")
f.close()

f = open("seq.fa", "r") 
print(f.read()) 

"""
komentarz do tego zadania:
w moim przypadku 'wyrzucane' przez program sekwencje zawierają takie same
znaki co zadane mu sekwencje. w przypadku podanego na ćwiczeniach wyniku
tej operacji, w każdej sekwencji brakowało jednego 'G'. nie wiem, czy u
mnie wyszło niepoprawnie, czy w wyniku z ćwiczeń pojawiła się literówka
"""

#zadanie 6

#tworze liste do zapisu z pliku
cale = []

u = open("C:\\Users\Paulina\Downloads\sekwencje","r")
for line in u:
    cale.append(line.strip())
u.close

name = []
seque = []

#zapisuje kolejne czesci listy calosc do list z nazwami i z sekwencjami
for i in range(len(cale)):
    if i%2 >0:
        seque.append(cale[i])
    else:
        name.append(cale[i][1:])

#tworze slownik
slo = dict(zip(name, seque))

print(slo)
    
#zadanie 7

#tworze sciezke dostepu, inaczej nie dalo sie otworzyc
import sys
sys.path.append(r'C:\\Users\Paulina\Desktop\Bioinformatyka\Natural language processing\ćwiczenia 2')
sys.path

#importuje modul
import paula as pc
dir(pc)

#wywoluje funkcje z modulu
print(pc.KTORE_PIERWSZE(21))
print(pc.REV_COM("ACTCGGTATT"))
print(pc.SUMA_POPRZEDNICH(7))

#zadanie 8

import math as mt
dir(mt)

#rozpisalam na zmienne, bo nie chcialam sie pomylic przy ostatecznym wyniku
a = mt.cos(mt.pi/6)
b = (mt.e)**7
c = mt.factorial(7)

print(a+b+c)#obliczam wartosc wyrazenia 