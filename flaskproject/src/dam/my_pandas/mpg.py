import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.cmm.service.menu import Menu

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 234 entries, 0 to 233
Data columns (total 11 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   manufacturer / 회사  234 non-null    object 
 1   model / 모델         234 non-null    object 
 2   displ / 배기량         234 non-null    float64
 3   year / 연식          234 non-null    int64  
 4   cyl / 실린더           234 non-null    int64  
 5   trans /차축         234 non-null    object 
 6   drv / 오토           234 non-null    object 
 7   cty /도시 연비           234 non-null    int64  
 8   hwy /고속 도로 연비           234 non-null    int64  
 9   fl / 연료          234 non-null    object 
 10  class / 차종        234 non-null    object 
dtypes: float64(1), int64(4), object(6)
memory usage: 20.2+ KB
None
'''
my_meta = {
    'manufacturer': '회사',
    'model': '모델',
    'displ': '배기량',
    'year': '연식',
    'cyl': '실린더',
    'trans': '차축',
    'drv': '오토',
    'cty': '도시 연비',
    'hwy': '고속 도로 연비',
    'fl': '연료',
    'class': '차종'
}


class Rkskekfk(object):
    def __init__(self, mpg):
        self.mpg = mpg
        self.mpg_copy = None  #
        self.count_test = None

    def menu_0(self, *params):  # 종료
        print('# ' + params[0])

    def menu_1(self, *params):  # head
        print('#' + params[0])
        print(self.mpg.head())

    def menu_2(self, *params):  # tail
        print('# ' + params[0])
        print(self.mpg.tail())

    def menu_3(self, *params):  # shape
        print('# ' + params[0])
        print(self.mpg.shape)

    def menu_4(self, *params):  # info
        print('# ' + params[0])
        print(self.mpg.info())

    def menu_5(self, *params):  # describe
        print('# ' + params[0])
        print(self.mpg.describe())

    def menu_6(self, *params):  # describe_include
        print('# ' + params[0])
        print(self.mpg.describe(include='all'))

    def menu_7(self, *params):  # 한글로 콜롬 바꾸기
        print('# ' + params[0])
        self.mpg_copy = self.mpg.rename(columns=my_meta)

    def menu_8(self, *params):  # 파생 변수 만들기
        print('# ' + params[0])
        sn = self.mpg_copy
        sn['총연비'] = (sn['도시 연비'] + sn['고속 도로 연비']) / 2
        sn['연비 테스트'] = np.where(sn['총연비'] >= 20, 'pass', 'fail')
        self.mpg_copy = sn
        print(self.mpg_copy.columns)
        print(self.mpg_copy.head())

    def menu_9(self, *params):  # 빈도
        print('# ' + params[0])
        self.menu_8('메뉴8에서 받아오기')
        t = self.mpg_copy
        self.count_test = t['연비 테스트'].value_counts()
        print(self.count_test)

    def draw_freq_bar_graph(self):  # No.10
        self.menu_9('메뉴9에서 받아오기')
        self.count_test.plot.bar(rot=0)
        plt.savefig('./save/draw_freq_bar_graph.png')

    ##############################################################################################0

    def menu_10(self, *params):  # 1204
        print('# ' + params[0])
        total4 = 0
        total5 = 0
        count4 = 0
        count5 = 0
        for j, i in enumerate(self.mpg['displ']):
            if i <= 4:
                total4 = total4 + self.mpg['hwy'][j]
                count4 = count4 + 1
            elif i >= 5:
                total5 = total5 + self.mpg['hwy'][j]
                count5 = count5 + 1
        avg4 = total4 / count4
        avg5 = total5 / count5
        if avg4 > avg5:
            print('배기량이 4 이하인 자동차의 연비 평균이 더 높습니다')
        elif avg4 < avg5:
            print('배기량이 5 이상인 자동차의 연비 평균이 더 높습니다')
        else:
            print('연비 평균이 같습니다')

    def menu_11(self, *params):  # 1204
        print('# ' + params[0])
        totalaudi = 0
        totaltoyota = 0
        countaudi = 0
        counttoyota = 0
        for j, i in enumerate(self.mpg['manufacturer']):  # numpy.mean(리스트 or 튜플)
            if i == 'audi':
                totalaudi = totalaudi + self.mpg['cty'][j]
                countaudi = countaudi + 1
            elif i == 'toyota':
                totaltoyota = totaltoyota + self.mpg['cty'][j]
                counttoyota = counttoyota + 1
        avgaudi = totalaudi / countaudi
        avgtoyota = totaltoyota / counttoyota
        if avgaudi > avgtoyota:
            print('audi 자동차의 도시 연비 평균이 더 높습니다')
        elif avgaudi < avgtoyota:
            print('toyota 자동차의 도시 연비 평균이 더 높습니다')
        else:
            print('도시 연비 평균이 같습니다')

    def menu_12(self, *params):  # 1204
        print('# ' + params[0])
        total = 0
        count = 0
        for j, i in enumerate(self.mpg['manufacturer']):
            if i == 'chevrolet' or 'ford' or 'honda':
                total = total + self.mpg['hwy'][j]
                count = count + 1
        avg = total / count
        print(f"고속도로 연비 평균 : {avg}")

    def menu_13(self, *params):
        print('# ' + params[0])
        new_mpg = self.mpg[['model', 'cty']]
        print(new_mpg)

    def menu_14(self, *params):  # 1204
        print('# ' + params[0])
        totalsuv = 0
        totalcompact = 0
        countsuv = 0
        countcompact = 0
        for j, i in enumerate(self.mpg['manufacturer']):
            if i == 'audi':
                totalsuv = totalsuv + self.mpg['cty'][j]
                countsuv = countsuv + 1
            elif i == 'toyota':
                totalcompact = totalcompact + self.mpg['cty'][j]
                countcompact = countcompact + 1
        avgsuv = totalsuv / countsuv
        avgcompact = totalcompact / countcompact
        if avgsuv > avgcompact:
            print('suv 자동차의 도시 연비 평균이 더 높습니다')
        elif avgsuv < avgcompact:
            print('compact 자동차의 도시 연비 평균이 더 높습니다')
        else:
            print('도시 연비 평균이 같습니다')

    def menu_15(self, *params):
        print('# ' + params[0])
        audi_df = self.mpg.query('manufacturer == "audi"')
        audi_df_sort = audi_df.sort_values('hwy', ascending=False)
        print(audi_df_sort.head(5))

    ######################################################################
    def menu_16(self, *params):
        print('# ' + params[0])
        self.mpg_copy = copy.deepcopy(self.mpg)
        self.mpg_copy['합산 연비 변수'] = self.mpg_copy['cty'] + self.mpg_copy['hwy']
        print(self.mpg_copy)

    def menu_17(self, *params):
        print('# ' + params[0])
        self.mpg_copy['평균 연비 변수'] = self.mpg_copy['합산 연비 변수'] / 2

    def menu_18(self, *params):
        print('# ' + params[0])
        self.mpg_copy.sort_values('평균 연비 변수', ascending=True).head(3)


menuls = ['종료', 'mpg 앞부분 확인', 'mpg 뒷부분 확인', '행, 열 출력', '데이터 속성 확인', '요약 통계량 출력',
          '문자 변수 요약 통계량 출력', '변수명 바꾸기', '파생변수 만들기', '빈도 확인하기', '4,5 연비 평균',
          '11', '12', '13', '14', '15', '16', '17', '18']

if __name__ == "__main__":
    ah = Rkskekfk(pd.read_csv("../../../static/data/dam/my_pandas/data/mpg.csv"))
    while 1:
        menu = Menu.print_menu(menuls)
        Menu.starprinter()
        if menu == '0':  # 종료
            ah.menu_0(menuls[0])
            Menu.starprinter()
            break
        elif menu == '1':  # mpg 앞부분 확인
            ah.menu_1(menuls[1])
            Menu.starprinter()

        elif menu == '2':  # mpg 뒷부분 확인
            ah.menu_2(menuls[2])
            Menu.starprinter()

        elif menu == '3':  # 행, 열 출력
            ah.menu_3(menuls[3])
            Menu.starprinter()

        elif menu == '4':  # 데이터 속성 확인
            ah.menu_4(menuls[4])
            Menu.starprinter()

        elif menu == '5':  # 요약 통계량 출력
            ah.menu_5(menuls[5])
            Menu.starprinter()

        elif menu == '6':  # 문자 변수 요약 통계량 출력
            ah.menu_6(menuls[6])
            Menu.starprinter()

        elif menu == '7':  # 변수명 바꾸기
            ah.menu_7(menuls[7])
            Menu.starprinter()

        elif menu == '8':  # 파생변수 만들기
            ah.menu_8(menuls[8])
            Menu.starprinter()

        elif menu == '9':  # 빈도 확인하기
            ah.menu_9(menuls[9])
            Menu.starprinter()

        elif menu == '10':  # 4,5 연비 평균
            ah.menu_10(menuls[10])
            Menu.starprinter()

        elif menu == '11':  # 4,5 연비 평균
            ah.menu_11(menuls[11])
            Menu.starprinter()

        elif menu == '12':  # 4,5 연비 평균
            ah.menu_12(menuls[12])
            Menu.starprinter()

        elif menu == '13':  # 4,5 연비 평균
            ah.menu_13(menuls[13])
            Menu.starprinter()

        elif menu == '14':  # 4,5 연비 평균
            ah.menu_14(menuls[14])
            Menu.starprinter()

        elif menu == '15':  # 4,5 연비 평균
            ah.menu_15(menuls[15])
            Menu.starprinter()

        elif menu == '16':  # 4,5 연비 평균
            ah.menu_16(menuls[16])
            Menu.starprinter()

        elif menu == '17':  # 4,5 연비 평균
            ah.menu_17(menuls[17])
            Menu.starprinter()

        elif menu == '18':  # 4,5 연비 평균
            ah.menu_18(menuls[18])
            Menu.starprinter()

        else:
            print('잘못된 메뉴')
