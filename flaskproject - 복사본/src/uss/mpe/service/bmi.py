from src.dam.ml import my_menu

BMI_MENUS = ["Exit", #0
                "Show Spec",#1
                "Save Police Position",#2.
                "Save CCTV Population",#3
                "Save Police Normalization",#4
                "Save US Unemployment Map",#5
                "Save Seoul Crime Map",#6
                ]

bmi_menu = {
    "1" : lambda t: t.show_spec(),
    "2" : lambda t: t.save_police_pos(),
    "3" : lambda t: t.save_cctv_pop(),
    "4" : lambda t: t.save_police_norm(),
    "5" : lambda t: t.save_us_unemployment_map(),
    "6" : lambda t: t.save_seoul_crime_map(),
}

class BMI(object):
    def __init__(self, name, height, weight, bmi, bimando):
        self.name = name
        self.height = height
        self.weight = weight
        self.bmi = bmi
        self.bimando = bimando

    @staticmethod
    def print_menu():
        print("메뉴\n0. 종료\n1. 회원 정보 입력\n2. 회원 정보 리스트\n3. 회원 정보 삭제")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def new_member():
        name = input("이름을 입력하시오 : ")
        height = int(input("키를 입력하시오 : "))
        weight = int(input("몸무게를 입력하시오 : "))
        bmi = weight / ((height / 100) * (height / 100))
        if bmi >= 35:
            bimando = '고도 비만'
        elif 35 > bmi >= 30:
            bimando = '중도 비만'
        elif 30 > bmi >= 25:
            bimando = '경도 비만'
        elif 25 > bmi >= 23:
            bimando = '과체중'
        elif 23 > bmi >= 18.5:
            bimando = '정상'
        else:
            bimando = '저체중'
        return BMI(name, height, weight, bmi, bimando)

    @staticmethod
    def get_list(member_ls):
        for i in member_ls:
            i.print_list()

    def print_list(self):
        print(f"이름 : {self.name} / 키 : {self.height} / 몸무게 : {self.weight} / bmi : {self.bmi} / 비만도 : {self.bimando}")

    @staticmethod
    def delete_member(member_ls, name):
        for i, j in enumerate(member_ls):
            if j.name == name:
                del member_ls[i]

if __name__ == '__main__':
    member_ls = []
    while 1:
        menu = BMI.print_menu()
        if menu == 1:
            print("1. 회원 정보 입력")
            member_ls.append(BMI.new_member())

        elif menu == 2:
            print("2. 회원 정보 리스트")
            BMI.get_list(member_ls)

        elif menu == 3:
            print("3. 회원 정보 삭제")
            BMI.delete_member(member_ls, input("탈퇴할 회원의 이름을 입력하시오 : "))

        elif menu == 0:
            print("0. 종료")
            break

        else:
            print("잘못된 메뉴를 입력하셨습니다.")


