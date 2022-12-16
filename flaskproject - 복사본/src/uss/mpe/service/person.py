"""
이름, 주민번호(950101-1), 주소를 입력받아서
회원명부 관리.

출력되는 결과는 다음과 같다.

### 자기소개어플 ###
************************************
이름:홍길동
나이:25살(만나이)
성별:남성
주소:서울
************************************
"""

class Person(object):
    def __init__(self, name, social_id_front, social_id_back, address, age, gender):
        self.name = name
        self.social_id_front = social_id_front
        self.social_id_back = social_id_back
        self.address = address
        self.age = age
        self.gender = gender

    @staticmethod
    def print_menu():
        print("1 . 회원 정보 입력\n2 . 회원 목록\n3 . 회원 탈퇴\n0 . 종료")
        return int(input("메뉴를 선택하시오 : "))

    @staticmethod
    def get_gender(social_id_back):
          if social_id_back == 1 or social_id_back == 3 or social_id_back == 9:
              return "남자"
          elif social_id_back == 2 or social_id_back == 4 or social_id_back == 0:
              return "여자"
          else:
              return "주민 등록 번호 뒷 자리를 잘못 입력하셨습니다"

    @staticmethod
    def get_age(social_id_front, social_id_back):
        newfront = 0
        if social_id_back == 9 or social_id_back == 0:
            newfront = social_id_front + 18000000  # 18991204
        elif social_id_back == 1 or social_id_back == 2:
            newfront = social_id_front + 19000000  # 19991204
        else:
            newfront = social_id_front + 20000000  # 209912104
        diff = 20221024 - newfront
        return diff // 10000

    @staticmethod
    def new_member(name, social_id_back, social_id_front, address, age, gender):
        name = input("이름을 입력하시오 : ")
        social_id_front = int(input("주민등록번호 앞 6자리를 입력하시오 : "))
        social_id_back = int(input("주민등록번호 뒤 1자리를 입력하시오 : "))
        address = input("주소를 입력하시오 : ")
        age = Person.get_age(social_id_front,social_id_back)
        gender = Person.get_gender(social_id_back)
        return Person(name, social_id_back, social_id_front, address, age, gender)

    @staticmethod
    def get_list(member_ls):
        for i in member_ls:
            i.print_ls

    def print_ls(self):
        print(f'이름 : {self.name} / 나이 : {self.age} / 성별 : {self.gender} / 주소 : {self.address}')

    @staticmethod
    def delete_member(member_ls, name):
        for i, j in enumerate(member_ls):
            if j.name == name:
                del member_ls[i]

if __name__ == '__main__':
    member_ls = []
    while 1:
        menu = Person.print_menu()
        if menu == 1:
            print("회원 정보 입력")
            member_ls.append(Person.new_member())

        elif menu == 2:
            print("회원 목록")
            Person.get_list(member_ls)

        elif menu == 3:
            print("회원 탈퇴")
            Person.delete_member(member_ls, input("탈퇴할 회원의 이름을 입력하시오 : "))

        elif menu == 0:
            print('종료')
            break

        else:
            print('잘못된 메뉴입니다. 다시 입력하시오')

