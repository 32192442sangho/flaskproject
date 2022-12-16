class Grade(object):
    def __init__(self, name, kor, eng, math, total, avg, grade):
        self.name = name
        self.kor = kor
        self.eng = eng
        self.math = math
        self.total = total
        self.avg = avg
        self.grade = grade

    @staticmethod
    def print_menu():
        print("메뉴\n1. 회원 정보 입력\n2. 회원 정보 리스트\n3. 회원 정보 삭제")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def new_member():
        name = input("이름을 입력하시오 : ")
        kor = int(input("국어 점수를 입력하시오 : "))
        eng = int(input("영어 점수를 입력하시오 : "))
        math = int(input("수학 점수를 입력하시오 : "))
        total = kor + eng + math
        avg = total / 3
        if avg >= 90:
            grade = "A"
        elif avg >= 80:
            grade = "B"
        elif avg >= 70:
            grade = "C"
        elif avg >= 60:
            grade = "D"
        elif avg >= 50:
            grade = "E"
        else:
            grade = "F"
        return Grade(name, kor, eng, math, total, avg, grade)

    @staticmethod
    def get_list(member_ls):
        for i in member_ls:
            i.print_list()

    def print_list(self):
        print(f"이름 : {self.name} / 국어 : {self.kor} / 영어 : {self.eng} / 수학 : {self.math} / 총점 : {self.total} / 평균 : {self.avg} / 학점 : {self.grade}")

    @staticmethod
    def delete_member(member_ls, name):
        for i, j in enumerate(member_ls):
            if j.name == name:
                del member_ls [i]

if __name__ == 'main':
    member_ls = []
    while 1:
        menu = Grade.print_menu()
        if menu == 1:
            print("회원 정보 입력")
            member_ls.append(Grade.new_member())

        elif menu == 2:
            print("회원 정보 리스트")
            Grade.get_list(member_ls)

        elif menu == 3:
            print("회원 정보 삭제")
            Grade.delete_member(member_ls, input("탈퇴할 회원의 이름을 입력하시오 : "))

        else:
            print("잘못된 메뉴 입력")

