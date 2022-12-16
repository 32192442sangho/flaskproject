"""
이름, 전번, 이메일, 주소 받기
연락처입력, 출력, 삭제 프로그램
인명은 여러명(->인스턴스가 여러개이다.) 저장 가능
"""
from src.uss.mpe.service import BMI


class Contact(object):
    def __init__(self, name, pn, email, ad):
        self.name = name
        self.pn = pn
        self.email = email
        self.ad = ad

    @staticmethod
    def print_menu():
        print("메뉴\n1. 회원 정보 입력\n2. 회원 정보 리스트\n3. 회원 정보 삭제")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def new_member():
        name = input("이름을 입력하시오 : ")
        pn = input("전화 번호를 입력하시오 : ")
        email = input("이메일을 입력하시오 : ")
        ad = input("주소를 입력하시오 : ")
        return BMI(name, pn, email, ad)

    @staticmethod
    def get_list(member_ls):
        for i in member_ls:
            i.print_list

    def print_list(self):
        print(f"이름 : {self.name} / 전화 번호 : {self.pn} / 이메일 : {self.email} / 주소 : {self.ad}")

    @staticmethod
    def delete_member(member_ls, name):
        for i, j in enumerate(member_ls):
            if j.name == name:
                del member_ls[i]

if __name__ == '__main__':
    member_ls = []
    while 1:
        menu = Contact.print_menu()
        if menu == 1:
            print("1. 회원 정보 입력")
            member_ls.append(Contact.new_member())

        elif menu == 2:
            print("2. 회원 정보 리스트")
            Contact.get_list(member_ls)

        elif menu == 3:
            print("3. 회원 정보 삭제")
            Contact.delete_member(member_ls, input("탈퇴할 회원의 이름을 입력하시오 : "))

        elif menu == 0:
            print("0. 종료")
            break

        else:
            print("잘못된 메뉴를 입력하셨습니다.")

