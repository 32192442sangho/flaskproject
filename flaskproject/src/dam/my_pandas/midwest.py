import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.cmm.service.menu import Menu

midwest = pd.read_csv("../../../static/data/dam/my_pandas/data/midwest.csv")
menu1_count = 0
def menu_0(*params):
    print('# ' + params[0])

def menu_1(*params):
    print('# ' + params[0])
    print(params[1].columns)
    params[1].rename(columns = {'poptotal' : 'total', 'popasian' : 'asian'},inplace=True)
    print(params[1].columns)

def menu_2(*params):        # 1 실행되어야함
    print('# ' + params[0])
    params[1]['전체 인구 대비 아시아 인구 백분율'] = params[1]['asian']/params[1]['total']*100

    tempdf = params[1]['전체 인구 대비 아시아 인구 백분율']
    params[1]['전체 인구 대비 아시아 인구 백분율'].index = params[1]['county']
    print(tempdf)
    tempdf.head().plot.bar(rot=0)

    plt.savefig('./save/전체 인구 대비 아시아 인구 백분율.png')

def menu_3(*params):       # 2 실행되어야함
    print('# ' + params[0])
    params[1]['avgupdown'] = np.where(params[1]['전체 인구 대비 아시아 인구 백분율'] >= sum(midwest['전체 인구 대비 아시아 인구 백분율'])/len(params[1]['전체 인구 대비 아시아 인구 백분율']), 'pass', 'fail')
    print(params[1])

def menu_4(*params):       # 3 실행되어야함
    print(params[1])
    params[1]['avgupdown'].value_counts().plot.bar(rot=0)
    plt.savefig('./save/countupdown.png')

menuls = ['종료', '변수명 바꾸기', '전체 인구 대비 아시아 인구 백분율 변수 만들기 & 전체 인구 대비 아시아 인구 백분율 빈도 확인하기', 'avgupdown 변수 만들기','avgupdown 빈도 확인하기']

if __name__ == "__main__":
    while 1:
        menu = Menu.print_menu(menuls)
        Menu.starprinter()
        if menu == '0':  # 종료
            menu_0(menuls[0])
            Menu.starprinter()
            break

        elif menu == '1':  # 변수명 바꾸기
            menu_1(menuls[1], midwest)
            Menu.starprinter()

        elif menu == '2':  # 파생변수 만들기 빈도 확인하기
            menu_2(menuls[2], midwest)
            Menu.starprinter()

        elif menu == '3':
            menu_3(menuls[3], midwest)
            Menu.starprinter()

        elif menu == '4':
            menu_4(menuls[4], midwest)
            Menu.starprinter()

        else:
            print('잘못된 메뉴')
