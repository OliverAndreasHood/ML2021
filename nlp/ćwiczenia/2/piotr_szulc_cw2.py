"""Zadania Piotr Szulc"""
#################################
print("\nZadanie1.:")
#Zadanie1: y = 3x^2 - 2
print(list(map(lambda x: 3*(x**2) - 2,[1,2,3,4,5])))

#################################
print("\nZadanie2.:")
#Zadanie2: ["ATTGC", "AGGC", "TTTGC"] => [2,0,3]
print(list(map(lambda s: s.count("T"), ["ATTGC", "AGGC", "TTTGC"])))

#################################
print("\nZadanie3.:")
#Zadanie3: parzyste z przedziału [1,40]
i = 40
print(list(filter(lambda p: p%2 == 0, range(1,i+1))))

#################################
print("\nZadanie4.:")
#Zadanie4: ["ATGCC", "TATC", "GGATGGGG"] =>  ["ATGCC", "GGATGGGG"]
print(list(filter(lambda seq: "ATG" in seq, ["ATGCC", "TATC", "GGATGGGG"])))

#################################
print("\nZadanie5.:")
#Zadanie5: seq1 i seq2 do pliku fasta
seq1 = "ATGGGC"
seq2 = "AGGGCT"

with open("z5.fasta", "w") as f:
    f.writelines(">seq1\n"+seq1+"\n")
    f.writelines(">seq2\n"+seq2+"\n")

#################################
print("\nZadanie6.:")
#Zadanie6: sekwencje => dictionary
x = []
n = []
s = []

with open("sekwencje.txt", "r") as f:
    for line in f:
        line = line[1:].rstrip()
        x.append(line)

for i in range(len(x)):
    if i%2 >0:
        s.append(x[i])
    else:
        n.append(x[i])
  
d = dict(zip(n,s))
print(d)

#################################
print("\nZadanie7.:")
#Zadanie7: własny moduł
import pscript as ps
x = 2/3
y = 5
z = 0.4

print(ps.przeciwna(y), ps.odwrotna(x), ps.multiple_reverse(z))

##################################
print("\nZadanie8.:")
#Zadanie8: oblicz y = cos(pi/6) + e**3 + 7!
import math as mt

print(mt.cos((mt.pi)/6) + (mt.e)**3 + mt.factorial(7))















