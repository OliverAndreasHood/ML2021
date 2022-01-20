# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:14:18 2020

@author: ExampleJohn


Liczby z przedziału
"""
#Napisz program, który wypisuje wszystkie liczby całkowite od -7 do 13 (funkcja range()).
#za pomocę pętli while

def wykrzykniki(s):
    i = s.count("!")
    s2 = s.replace("!","")
    w = "!"
    w2 = ""
    for x in range(3*i):
        w2+=w
    return s2 + w2

s = "Czesc!!"
print(wykrzykniki(s))