# -*- coding: utf-8 -*-

def KTORE_PIERWSZE(n):
    #zwraca liczby pierwsze =<n
    a = [] 
    for i in range(2, n+1): #wszystkie liczby naturalne od 2 do n
            for h in range(2,i): #sprawdzanie czy i ma dzielniki
                if i%h == 0: #likwidacja "pierwszych" dzielników
                    break
                else:
                    if h == i-1: #likwidacja kolejnych dzielników
                        a.append(i)
    return a


def REV_COM(x):
    #zwraca nić odwrotnie komplementarną
    
    com_code = {"A":"T", "T":"A", "G":"C", "C":"G"}
    rev_seq = '' 
   
    for i in x:
        rev_seq = com_code[i] + rev_seq 

    return rev_seq 


def SUMA_POPRZEDNICH(x):
    if x==1:
        return x
    elif x>1:
        n=0
        for i in range(1,x+1):
            n=n+i
        return n
    else:
        return "Nie jest liczbą naturalną"

