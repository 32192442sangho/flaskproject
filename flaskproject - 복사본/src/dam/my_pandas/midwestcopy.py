import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.cmm.service.menu import Menu


class Midwest(object):
    def __init__(self, midwest):
        self.midwest = midwest #pd.read_csv("./data/midwest.csv")


    def menu_0(self, *params):
        print('# ' + params[0])

    def menu_1(self, *params):
        print('# ' + params[0])
        print(self.midwest.columns)
        self.midwest.rename(columns = {'poptotal' : 'total', 'popasian' : 'asian'},inplace=True)
        print(self.midwest.columns)


    def menu_2(self, *params):        # 1 실행되어야함
        print('# ' + params[0])

        self.midwest['전체 인구 대비 아시아 인구 백분율'] = self.midwest['asian']/self.midwest['total']*100

        tempdf = self.midwest['전체 인구 대비 아시아 인구 백분율']
        tempdf.index = self.midwest['county']
        print(tempdf)
        tempdf.head().plot.bar(rot=0)

        plt.savefig('./save/전체 인구 대비 아시아 인구 백분율.png')

    def menu_3(self, *params):       # 2 실행되어야함
        print('# ' + params[0])
        ap100avg = sum(self.midwest['전체 인구 대비 아시아 인구 백분율'])/len(self.midwest['전체 인구 대비 아시아 인구 백분율'])
        self.midwest['avgupdown'] = np.where(self.midwest['전체 인구 대비 아시아 인구 백분율'] >= ap100avg, 'pass', 'fail')
        print(self.midwest)

    def menu_4(self, *params):       # 3 실행되어야함
        print('# '+ params[0])
        count_test = self.midwest['avgupdown'].value_counts()
        count_test.plot.bar(rot=0)
        plt.savefig('./save/countupdown.png')

menuls = ['종료', '변수명 바꾸기', '전체 인구 대비 아시아 인구 백분율 변수 만들기 & 전체 인구 대비 아시아 인구 백분율 빈도 확인하기', 'avgupdown 변수 만들기','avgupdown 빈도 확인하기']

if __name__ == "__main__":
    mw = Midwest(pd.read_csv("../../../static/data/dam/my_pandas/data/midwest.csv"))
    while 1:
        menu = Menu.print_menu(menuls)
        Menu.starprinter()

        if menu == '0':  # 종료
            mw.menu_0(menuls[0])
            Menu.starprinter()
            break

        elif menu == '1':  # 변수명 바꾸기
            mw.menu_1(menuls[1])
            Menu.starprinter()

        elif menu == '2':  # 파생변수 만들기 빈도 확인하기
            mw.menu_2(menuls[2])
            Menu.starprinter()

        elif menu == '3':
            mw.menu_3(menuls[3])
            Menu.starprinter()

        elif menu == '4':
            mw.menu_4(menuls[4])
            Menu.starprinter()

        else:
            print('잘못된 메뉴')
