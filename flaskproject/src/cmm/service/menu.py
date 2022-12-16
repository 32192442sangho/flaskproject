class Menu(object):
    def __init__(self):
        pass

    @staticmethod
    def print_menu(list_of_menu):
        for i, j in enumerate(list_of_menu):
            print(f"{i}. {j}")
        return input("메뉴 선택 : ")

    @staticmethod
    def starprinter():
        print('*'*200)

