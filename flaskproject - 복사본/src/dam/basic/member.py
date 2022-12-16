"""
아이디 비밀번호 이름 받아서
회원가입 목록 탈퇴하는 프로그램 개발
"""
class Member (object):
    def __init__(self, id, pw, name):
        self.id = id
        self.pw = pw
        self.name = name

    @staticmethod
    def print_menu():
        print("1 . 회원 가입\n2 . 회원 목록\n3 . 회원 탈퇴\n0 . 종료")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def new_member():
        return Member(input("아이디를 입력하시오 : "), input("비밀 번호를 입력하시오 : "), input("이름을 입력하시오 : "))

    @staticmethod
    def get_list(member_ls):
        [i.print_list() for i in member_ls]

    def print_list(self):
        print(f"ID : {self.id} / PW : {self.pw} / 이름 : {self.name}")

    @staticmethod
    def delete_member(member_ls,name):
        for i, j in enumerate(member_ls):
            if j.name == name:
                del member_ls[i]

if __name__ == '__main__':
    member_ls = []
    while 1:
        menu = Member.print_menu()
        if menu == 1:
            print("회원 가입")
            member_ls.append(Member.new_member())

        elif menu == 2:
            print("회원 목록")
            Member.get_list(member_ls)

        elif menu == 3:
            print("회원 탈퇴")
            Member.delete_member(member_ls, input("탈퇴할 회원의 이름을 입력하시오 : "))

        elif menu == 0:
            print('종료')
            break

        else:
            print('잘못된 메뉴입니다. 다시 입력하시오')



