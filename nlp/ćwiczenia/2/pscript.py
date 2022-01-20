def przeciwna(x):
    """zwraca wartosc przeciwną do x"""
    if x != 0:
        x= -x
        return x
    else:
        return x
    
def odwrotna(x):
    """zwraca odwrotnosc x"""
    if x != 0:
        x = 1/x
        return x
    else:
        return x
    
def multiple_reverse(x):
    """dla x != 0 zwraca iloraz x i odwrotnosci x"""
    if x != 0:
        x = x*(1/x)
        return x
    else:
        return x
    