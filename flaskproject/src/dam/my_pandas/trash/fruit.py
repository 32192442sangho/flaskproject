import string

import numpy as np
import pandas as pd


def new_fruits_df():
    schema = ['제품', '가격', '판매량']
    fruit = ['사과', '딸기', '수박']
    cost = [1800, 1500, 3000]
    sellrate = ['24', '38', '13']

    value = (fruit, cost, sellrate)

    dc = {j: value[i] for i, j in enumerate(schema)}
    print(f"{dc}\n")

    fruit_df = pd.DataFrame(dc)  # , orient='index'       .from_dict

    print(fruit_df)
    print('\n')
    print(fruit_df['가격'])
    print('\n')
    total = sum(fruit_df['가격'])
    avg = total // len(fruit_df['가격'])
    print(f"평균 : {avg}원")


def new_num_2d():
    num_2d = pd.DataFrame(np.array([list(range(10*i-9, 10*i+1)) for i in range(1,4)]), columns=map(chr, range(97, 107)))
    # coluns = list(string.ascii_lowercase)[0:10]  columns=map(chr, range(97, 107))
    print(num_2d)


if __name__ == '__main__':
    menuls = ['end', 'new_fruits', 'new_num']

    while 1:
        [print(f"{j}. {i}") for j, i in enumerate(menuls)]
        menu = int(input('메뉴 : '))
        if menu == 0:
            print(menuls[0])
            break
        elif menu == 1:
            print(menuls[1])
            new_fruits_df()
        elif menu == 2:
            print(menuls[2])
            new_num_2d()
        else:
            print('잘못된 메뉴')
