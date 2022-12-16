from dataclasses import dataclass

@dataclass
class OOP:
    x = 30

    def foo(self):
        x = self.x
        print("OOP 출력: "+str(x))
x = 100
def calc():
    a = 3
    b = 5
    total = 0
    def mul_add(x):
        nonlocal total
        c = 0
        total = total + a * x + b
        c = c + a * x + b
        print(total)
        print(c)
        print('*'*10)
    return mul_add

def counter():

    i = 0
    def count():
        nonlocal i
        i = i + 1
        return i
    return count



if __name__ == '__main__':
    c = counter()
    for i in range(10):
        print(c(), end=' ')
