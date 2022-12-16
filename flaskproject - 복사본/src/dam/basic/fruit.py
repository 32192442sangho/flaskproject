"""
과일 판매상에서 메뉴를 진역하는 어플 제작
입력값 없음
출력결과
### 과일 번호표 ###
*****************************************
1번 과일 : 바나나
2번 과일 : 사과
3번 과일 : 망고
*****************************************

"""
class Fruit(object):
    def __init__(self, fruit_name ,cost):
        self.fruit_name = fruit_name
        self.cost = cost

    @staticmethod
    def print_menu():
        print("메뉴\n1. 과일 입력\n2. 과일 리스트\n3. 과일 삭제")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def new_fruit():
        return Fruit(input("과일 이름을 입력하시오 : "), input("가격을 입력하시오 : "))

    @staticmethod
    def get_list(fruit_ls):
        for i in fruit_ls:
            i.print_list

    def print_list(self):
        print(f"과일 이름 : {self.fruit_name} / 과일 가격 : {self.cost}")

    @staticmethod
    def delete_fruit(fruit_ls, name):
        for i, j in enumerate(fruit_ls):
            if j.fruit_name == name:
                del fruit_ls[i]

if __name__ == '__main__':
    fruit_ls = []
    while 1:
        menu = Fruit.print_menu()
        if menu == 1:
            print("1. 과일 입력")
            fruit_ls.append(Fruit.new_fruit())
        elif menu == 2:
            print("2. 과일 리스트")
            Fruit.get_list(fruit_ls)
        elif menu == 3:
            print("3. 과일 삭제")
            Fruit.delete_fruit(fruit_ls, input("지울 과일의 이름을 입력하시오 : "))
        elif menu == 0:
            print("0. 종료")
            break
        else:
            print("잘못된 메뉴를 선택하셨습니다.")
