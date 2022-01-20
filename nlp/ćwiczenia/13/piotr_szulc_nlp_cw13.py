"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw12"""

#################################
print("\nZadanie 1.:")
# Zadanie1: Wyjaśnij skąd wzięła się wartość funkcji kosztu 
# (wyznacz ją poprzez bezpośrednie obliczenia).

import numpy as np
print("In [28]: criterion = nn.CrossEntropyLoss()\n=> typ funkcji kosztu: Entropia Krzyżowa")

out = [ 0.0879, -0.2912,  0.0025, -0.1824,  0.1191,  0.0798,  0.0715,  0.1983,
       -0.0485,  0.1410, -0.1918,  0.1621,  0.1636, -0.1932, -0.0672, -0.2396,
       -0.0008,  0.0446,  0.2156,  0.0806, -0.1410,  0.2470]

print("\nWyniki przewidywania modelu:\nout =", out)

y = [0,0,0,0,0,
     0,0,0,0,0,
     0,0,0,0,1,
     0,0,0,0,0,
     0,0]

print("\nPrzynależnosć do klasy:\ny =", y)

ezi = [np.power(np.e, zi) for zi in out]

print("\nWyznaczone exponenty z przewidywania modelu:\nezi =", ezi)

y_pred = ezi/sum(ezi)

print("\nWyznaczenie wartosci y_pred w funkcji exponenty:\ny_pred =", y_pred)

CE = -sum([y[i]*np.log(y_pred[i]) for i in range(len(y))])

print("\nWyliczona wartosc funkcji kosztu:\n>",CE)
print("co zgadza się z out[34]:\n> tensor(3.1819, grad_fn=<NllLossBackward>)")


#################################
print("\n\nZadanie 2.:")
# Zadanie2: Dlaczego liczba iteracji wynosiła ok 2600 
# skoro zadaliśmy liczbę epok na 200?
l = [[131, 130, 126, 125, 123, 121, 119, 119, 118, 116, 116, 115, 113, 113, 109, 109],
     [55, 55, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 53, 53, 52, 52],
     [46, 46, 46, 46, 45, 45, 45, 45, 45, 45, 44, 44, 44, 44, 44, 44],
     [41, 41, 41, 41, 40, 40, 40, 40, 40, 38, 38, 38, 38, 38, 37, 37],
     [35, 34, 34, 34, 33, 33, 33, 33, 32, 32, 31, 31, 30, 30, 30, 30],
     [172, 170, 168, 168, 166, 165, 164, 161, 160, 160, 158, 157, 156, 155, 152, 150],
     [86, 85, 84, 84, 83, 83, 83, 83, 82, 82, 82, 82, 81, 81, 81, 80],
     [247, 247, 235],
     [63, 63, 63, 62, 62, 61, 61, 61, 61, 60, 60, 60, 60, 59, 59, 59],
     [24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22],
     [28, 27, 27, 27, 26, 26, 26, 26, 25, 25, 25, 24, 24, 24, 24, 24],
     [51, 51, 51, 51, 50, 50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49],
     [150, 149, 148, 146, 144, 142, 142, 141, 141, 140, 137, 137, 136, 135, 133, 133],
     [12, 12, 12, 11, 11, 11,  9,  9,  9,  9,  8,  8,  8,  7,  6,  4],
     [58, 58, 58, 58, 58, 57, 57, 57, 57, 56, 56, 56, 56, 55, 55, 55],
     [37, 37, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 35, 35, 35, 35],
     [229, 225, 216, 207, 200, 196, 195, 194, 193, 186, 185, 178, 175, 174, 173, 173],
     [74, 74, 73, 72, 71, 71, 71, 71, 71, 70, 69, 69, 69, 68, 68, 68],
     [30, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 29, 28, 28],
     [96, 96, 95, 95, 95, 95, 94, 94, 93, 90, 90, 89, 89, 89, 87, 86],
     [79, 79, 78, 78, 77, 77, 77, 77, 77, 76, 76, 76, 76, 75, 75, 75],
     [68, 68, 68, 67, 66, 66, 65, 65, 65, 65, 64, 64, 64, 64, 63, 63],
     [108, 107, 107, 106, 104, 103, 102, 102, 102, 101, 101, 101,  98,  98, 97,  97],
     [44, 44, 44, 43, 43, 43, 43, 43, 43, 42, 42, 42, 42, 42, 42, 42],
     [49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 47, 46, 46],[21, 21, 21, 20, 20, 20, 20, 20, 20, 18, 18, 17, 17, 16, 15, 12]]

print("Ilosć uzyskanych tensorów dla batch_size = 16: ",len(l))

print("\nDla batch_size = 16, otrzymalismy 26 tensorów, \
na których trenujemy model po 200 razy każdy. \
W tym wypadku otrzymalibysmy 5200 iteracji.\n\
W In [94], batch_size = 32 więc otrzymamy 13 tensorów. \
W tym przypadku otrzymamy 13*200 iteracji czyli 2600.")

#################################
print("\n\nZadanie 3.:")
# Zadanie3: Poniżej wygenerowano dwie sekwencje (losowe cytaty) 
# - jeden z temperaturą 0.6, drugi z 1.5. 
# Który pochodzi z którego losowania? Dlaczego?

print("In [126]:\tprint(sample_sequence(model, temperature=T))  #T=?\n\
Out [126]:\tWhat we make always seeds now\n\n\
In [104]:\tprint(sample_sequence(model, temperature=T))  #T=?\n\
Out [104]:\tThomer bullumiber we-dne blid a nee find Gached\n")

print("Wartosci T>1 powodują rozmycie generowanego tekstu.\
Rozwarzamy wartosci T = 0.6 i T = 1.5 .\
Oznacza to, że zdanie mające bardziej nielogiczną składnie,\
będzie odpowiadać wartosci T = 1.5 czyli będzie to zdanie\
wygenerowane w komórce In[104].\
W komórce In[126], drogą wykluczeń wartosć T = 0.6,\
ale też ponieważ wygenerowane zdanie ma sens logiczny i składniowy.")































