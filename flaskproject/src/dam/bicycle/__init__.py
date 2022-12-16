from src.dam.bicycle.models import Bicycle_Model
from src.dam.bicycle.views import Bicycle_Controller
from src.utl import Menu

api = Bicycle_Controller()
while 1:
    menu = Menu.print_menu(["종료", "시각화", "모델링", "머신 러닝", "배포"])
    if menu == '0':
        print('종료')
        break

    elif menu == '1':
        model = Bicycle_Model()
        a = model.new_model(f'{input("파일이름 : ")}.csv')
        print('##시각화##')
        print(f'헤드{a.head()}')
        print(f'콜롬{a.columns}')
        print(f'헤드{a.head()}')
        print(f'널 갯수{a.isnull().sum()}')

    elif menu == '2':
        print('모델링')
    elif menu == '3':
        print('머신 러닝')
    elif menu == '4':
        print('배포')
    else:
        print("해당 메뉴 없음")