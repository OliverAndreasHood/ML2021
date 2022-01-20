"""Zadania Piotr Szulc"""
"""#################################
print("\nZadanie1.:")
#Zadanie1: 10 elementów iloczyn trzech poprzednich
x = [0.5,1,2]

while len(x) < 10:
    x.append(x[-1]*x[-2]*x[-3])    

print(x)

#################################
print("\nZadanie2.:")
#Zadanie2: suma poprzednich
def SUMA_POPRZEDNICH(x):
    s = 0
    while x != 0:
        s+=x
        x-=1
    return s

print(SUMA_POPRZEDNICH(4))

#################################
print("\nZadanie3.:")
#Zadanie3: rev com
def REV_COM(seq):
    com_code = {"A":"T", "T":"A", "G":"C", "C":"G"}
    com_seq = ''
    rev_seq = ''
    for elem in seq:
        com_seq = com_seq + com_code[elem]
    
    for i in range(1,len(com_seq)+1):
        rev_seq = rev_seq + com_seq[-i]
    
    return rev_seq

print(REV_COM("ATGGC"))

#################################
print("\nZadanie4.:")
#Zadanie4: ktore_pierwsze
def KTORE_PIERWSZE(n):
    a = []
    
    for i in range(1, n+1):
        c = 0
        for x in range(1, i):
            if i%x == 0: 
                c+=1
                #print("weszło",i,x)
        if c == 1:
            a.append(i)    
    
    return a

print(KTORE_PIERWSZE(20))
"""
#################################
print("\nZadanie5.:")
#Zadanie5: Najdłuższa podsekwencja
#def NAJDLUZSZA_PODSEKWENCA(seq1, seq2):
    


def Compare_sequences(seq1, seq2):
    
    """
    Funkcja Compare_sequences dla 2 sekwencji
    nukleotydowych tej samej dlugosci zwraca
    liste z informacją na ktorych pozycjach
    jest ten sam nt [1] oraz rozny [-2]
    """
    
    N1 = len(seq1)
    N2 = len(seq2)
    
    if N1 == N2:    #jezeli sekwencje mają tę samą dlugosc
        y = []    #tworze pusta liste
        for i in range(N1): #przechodze kolejne indeksy pozycji
            if seq1[i]==seq2[i]:  #jezeli sekwencje mają ten sam nt na kolejnej pozycji
                y.append(1) #wstaw 1
            else:
                y.append(-2) #w przeciwnym przypadku wstaw -2
        return y #zwroc y
    
    else:    #jezeli sekwencje nie są tych samych dlugosci
        return "Sekwencje są różnych długości"    


seq1 = "ATGGTGAGCCACGACTGGTCA"
seq2 = "TGCGGCACATGATCGTGTAGA"

def SCORE(seq1, seq2):
    
    """
    Funkcja SCORE liczy
    podobieństwo pomiędzy
    dwoma sekwencjami
    """
    
    N1 = len(seq1)
    N2 = len(seq2)
    if N1 ==N2:
        S = sum(Compare_sequences(seq1, seq2))  #sumuj elementy z listy zwracanej przez Compare_sequences
        return S
    else:
        return "Sekwencje mają różne długości"
    
print(SCORE(seq1, seq2))
    
    
    
    
    
    
    
    
    
    
    
    
    
