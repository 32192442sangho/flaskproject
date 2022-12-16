class Calc(object):
    def __init__(self, name, num1, op, num2, result):
        self.name = name
        self.num1 = num1
        self.op = op
        self.num2 = num2
        self.result = result

    @staticmethod
    def print_menu():
        print("메뉴\n0. 종료\n1. 계산\n2. 기록보기\n3. 기록삭제")
        return int(input("메뉴를 선택 : "))

    @staticmethod
    def calc():
        num1 = int(input("숫자를 입력하시오 : "))
        op = input("기호를 입력하시오 : ")
        num2 = int(input("숫자를 입력하시오 : "))
        if op == "+":
            result = num1 + num2
        elif op == "-":
            result = num1 - num2
        elif op == "/":
            result = num1 / num2
        elif op == "//":
            result = num1 // num2
        elif op == "%":
            result = num1 % num2
        elif op == "*":
            result = num1 * num2
        return Calc(input("계산자 입력 : "), num1, op, num2, result)

    @staticmethod
    def get_history(calc_history):
        for i in calc_history:
            i.print_history()

    def print_history(self):
        print(f"계산자 : {self.name} / {self.num1} {self.op} {self.num2} = {self.result}")

    @staticmethod
    def delete_history(history, name):
        for i, j in enumerate(history):
            if j.name == name:
                del history[i]

if __name__ == '__main__':
    calc_history = []
    while 1:
        menu = Calc.print_menu()
        if menu == 1:
            print("계산")
            calc_history.append(Calc.calc())
        elif menu == 2:
            print("기록보기")
            Calc.get_history(calc_history)

        elif menu == 3:
            print("제거")
            Calc.delete_history(calc_history, input("계산자 입력 : "))

        elif menu == 0:
            print("종료")
            break
        else:
            print("잘못 입력하셨습니다")

