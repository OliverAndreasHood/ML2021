"""Zadania Piotr Szulc - przetwarzanie języka naturalnego cw8"""

#################################
print("\nZadanie 1.:")
# Zdefiniuj dowolne dwie macierze kwadratowe  A  i  B  
# o wymiarach (3,3) (trzy kolumny i trzy wiersze). 
# Przekonaj się, że mnożenie macierzy nie jest przemienne, 
# tj.  AB  i  BA  to różne macierze.

import numpy as np

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[2,3,4],[5,6,7],[8,9,0]])

X1 = np.dot(A,B)
X2 = np.dot(B,A)
print("Macierz X1 = AB:")
print(X1)
print("Macierz X2 = BA:")
print(X2)

#################################
print("\nZadanie 2.:")
# Wyznacz macierz  B4+AB+6B−1A , 
# gdzie  A  i  B  zdefiniowane w poprzednim zadaniu.

from numpy.linalg import inv
from numpy.linalg import matrix_power as mp

B1 = inv(B)
X3 = mp(B,4) + X1 +6*(np.dot(B1,A))
print("Macierz X3 = B^4 + A*B + 6*B^(-1)*A")
print(X3)

#################################
print("\nZadanie 3.:")
# Zdefiniuj dane uczące  X=np.array([1,2,3,4,5])  oraz  
# Y=np.array([−1,3,13,29,51]) . Zakładamy, że pomiędzy  
# X  i  Y  istnieje zależność  Y=aX2+bX+c . 
# Zbuduj model przewidujący wartość parametrów  a,b,c  
# w oparciu o  X  i  Y . Następnie wyznacz przewidywaną
# wartość dla  x=6 .

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([-1, 3, 13, 29, 51])

a = 1.0
b = 1.0
c = 1.0

def forward(x):
    return a*x**2 + b*x + c

def loss(y, y_pred):
    return np.mean(((y_pred - y)**2))

def gradient(x, y, y_pred):
    return np.mean(2*(x**2)*(y_pred - y)), np.mean(2*x*(y_pred - y)), np.mean(2*(y_pred - y))


print(f'Ile wynosi f(6): {forward(6):.3f}')

learning_rate = 0.001
n_iters = 1000

for epoch in range(n_iters):
    y_pred = forward(X) #liczymy ile wynoszą przewidywane wartości dla X
    l = loss(Y, y_pred) #wyznaczamy wartość funkcji kosztu
    dl = gradient(X, Y, y_pred) #liczymy gradient
    a -= learning_rate * dl[0]    #aktualizujemy parametr a
    b -= learning_rate * dl[1]    ##aktualizujemy parametr b
    c -= learning_rate * dl[2]    ###aktualizujemy parametr c
    
    if epoch % 200 == 0: #Co 200 przejscie zobaczmy parametry i wartosc funkcji kosztu
        print(f'epoch {epoch+1}: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, loss = {l:.8f}')
  
print(f'Ile wynosi f(6) po trenowaniu: {forward(6):.3f}')

#################################
print("\nZadanie 4.:")

X=np.array([-3,-2,-1,0,1,2,3], dtype=np.float32)
Y=np.array([27,-32,-13,0,-5,32,243])

def res_par():
    return 0.0, 0.0, 0.0, 0.0, 0.0
a,b,c,d,e = res_par()
    
def forward2(x):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

def loss2(y, y_pred):
    return np.mean(((y_pred - y)**2))

def gradient2(x, y, y_pred):
    return np.mean(2*(x**4)*(y_pred - y)), np.mean(2*(x**3)*(y_pred - y)), np.mean(2*(x**2)*(y_pred - y)), np.mean(2*x*(y_pred - y)), np.mean(2*(y_pred - y))


def model(a,b,c,d,e,lr,n_iters):
    for epoch in range(n_iters):
        y_pred = forward2(X) #liczymy ile wynoszą przewidywane wartości dla X
        l = loss2(Y, y_pred) #wyznaczamy wartość funkcji kosztu
        dl = gradient2(X, Y, y_pred) #liczymy gradient
        a -= lr * dl[0]
        b -= lr * dl[1]
        c -= lr * dl[2]
        d -= lr * dl[3]
        e -= lr * dl[4]
        if epoch % (n_iters/5) == 0: #Co 5-tą czesc przejscia zobaczmy parametry i wartosc funkcji kosztu
            print(f'epoch {epoch+1}: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, d = {d:.3f}, e = {e:.3f}, loss = {l:.8f}')


print("\n > 1st* => lr = 0.01 n_iters = 10")
print(f'Ile wynosi f(4): {forward(4):.3f}')
model(0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 10)
print(f'Ile wynosi f(4) po trenowaniu: {forward2(4):.3f}\n')

print("\n > 2nd* => lr = 0.0001 n_iters = 10")
print(f'Ile wynosi f(4): {forward(4):.3f}')
model(0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 10)
print(f'Ile wynosi f(4) po trenowaniu: {forward2(4):.3f}\n')

print("\n > 3th* => lr = 0.0001 n_iters = 10000")
print(f'Ile wynosi f(4): {forward(4):.3f}')
model(0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 10000)
print(f'Ile wynosi f(4) po trenowaniu: {forward2(4):.3f}\n')

print("\n > 4th* => lr = 0.0001 n_iters = 10000")
print(f'Ile wynosi f(4): {forward(4):.3f}')
model(2.7, 4.2, -11.9, 0.1, 0.0, 0.0001, 10000)
print(f'Ile wynosi f(4) po trenowaniu: {forward2(4):.3f}\n')

#################################
print("\nZadanie 5.:")
#Utwórz dwa tensory jednowymiarowe  x=tensor([1,2,3,4,5,6])  
#oraz  y=tensor([7,8,3,−1.,−2,2]) . Następnie wyznacz nowy 
#tensor wg przepisu  z=x3+2y . Utwórz dwa nowe tensory 
#(2D - o wymiarach 2 na 3 oraz 3D - o wymiarach 2,1,3 odpowiednio) 
#poprzez zmianę wymiaru  z .
import torch

x = torch.tensor([1,2,3,4,5,6])
y = torch.tensor([7,8,3,-1,-2,2])
z = x**3+2*y

z2D = z.view(2, 3)
z3D = z.view(2,1,3)
print("Wynik 2D:")
print(z2D)
print("\nWynik 3D:")
print(z3D)


#################################
print("\nZadanie 6. i 7.:")
print("6.:\nx = torch.tensor([-3.0,4,1], requires_grad=True) \t=> utowrzenie tensora \'x\'\
      \nz = x**3 \t=> podniesienie wartosci tensora \'x\' do potęgi 3; zapisanie jako \'z\'\
      \nr = z.sum() \t=> zsumowanie wartoci tensora \'z\'; zapisanie jako \'r\'\
      \nprint(r) \t=> wypisanie sumy \'r\'\
      \nr.backward() \t=> wsteczna propagacja sumy \'r\'\
      \nprint(x.grad) \t=> wypisanie przybliżonych wartosci propagacji dla \'x\'")


print("\n\n7.:\nx = torch.tensor([-3.0,4,1], requires_grad=True) \t=> utworzenie tensora \'x\'\
      \ny = torch.tensor([2.0,0,-1], requires_grad=True) \t=> utworzenie tensora \'y\'\
      \nz = x**2*y \t=> podniesienie tensora \'x\' do potęgi 2 i pomnożenie przez tensor \'y\'; zapisanie jako \'z\'\
      \nr = z.sum() \t=> zsumowanie wartoci tensora \'z\'; zapisanie jako \'r\'\
      \nprint(r) \t=> wypisanie sumy \'r\'\
      \nr.backward() \t=> wsteczna propagacja sumy \'r\'\
      \nprint(x.grad) \t=> wypisanie przybliżonych wartosci propagacji dla \'x\'")

print("\n\nr.backward() => zmieniamy \'z\' i wyznaczamy \'r\' propagacją wsteczną - czyli chcemy wiedziec jakie liczby się zmienily w \'z\'\
      \n\nprint(x.grad) => pochodna \'x\' po \'z\'; podaje nam przybliżone wartości początkowe których wynikiem była suma \'r\'.")




















