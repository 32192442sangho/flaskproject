
import matplotlib.pyplot as plt #이걸로 시각화
import seaborn as sns

from src.cmm.service.dataset import Dataset
import numpy as np
import pandas as pd

class Titanic_Model(object):
    dataset = Dataset()

    def __init__(self):
        pass

    def __str__(self):
        b = self.new_model(self.dataset.fname)
        return f"타입 \n{type(b)}\n\n" \
               f"콜롬 \n{b.columns}\n\n" \
               f"헤드\n{b.head()}\n\n" \
               f"널 갯수 \n{b.isnull().sum()}\n\n"

    def preprocess(self):  # 클래스 내부 데이터 처리
        pass

    def new_model(self, fname) -> object:
        this = self.dataset
          # context = 경로
        this.fname = fname
        df = pd.read_csv('C:/Users/gpark/PycharmProjects/flaskProject/static/data/dam/titanic/data/' + this.fname)
        return df

    @staticmethod  # 클래스 외부 데이터 처리
    def create_train(this) -> object:  # 데이터 인스턴스화
        return this.train.drop('Survived', axis=1)

    @staticmethod
    def create_label(this):  # 테스트용은 보통 label이라고 부름
        return this.train['Survived']

    @staticmethod
    def drop_features(this, *feature) -> object:  # *->리스트 담는 과정 생략
        for i in feature:
            this.train = this.train.drop(i, axis=1)
            this.test = this.test.drop(i, axis=1)
        return this

    """
    @staticmethod
    def pclass_ordinal(this)-> object:   #1,2,3 =>순서있음
        train = this.train
        test = this.test

        return this
    """

    @staticmethod
    def age_ordinal(this) -> object:  # 10대 20대 30대

        for i in [this.train, this.test]:
            i["Age"] = i["Age"].fillna(-0.5)  # -0.5는 없는 나이니까 넣어서 연령미상처리
        bins = [-1, 0, 5, 12, 18, 24, 35, 68, np.inf]  # np.inf -> 무한
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        age_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6,
                       'Senior': 7}
        for i in [this.train, this.test]:
            i['AgeGroup'] = pd.cut(i['Age'], bins, labels=labels)
            i['AgeGroup'] = i['AgeGroup'].map(age_mapping)
        return this

    @staticmethod
    def sex_nominal(this) -> object:  # male -> 0, female -> 1 /앞 자연어 뒤 기계어 -> 딕셔너리
        for i in [this.train, this.test]:
            i['Gender'] = i['Sex'].map({'male': 0, 'female': 1})  # .map?
        return this

    @staticmethod
    def fare_ordinal(this) -> object:  # 비쌈, 보통, 저렴 // pd.qcut()
        for i in [this.train, this.test]:
            i['FareBand'] = pd.qcut(i['Fare'], 4, labels=[1, 2, 3, 4])
        return this

    @staticmethod
    def embarked_nominal(this) -> object:  # 승선항구 S,C,Q
        # {'S': 0, 'C': 1, 'Q' : 2}
        this.train = this.train.fillna({'Embarked': 'S'})
        this.test = this.test.fillna({'Embarked': 'S'})
        for i in [this.train, this.test]:
            i['NewEmbarked'] = i['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        return this

    @staticmethod
    def title_nominal(this) -> object:
        combine = [this.train, this.test]
        for i in combine:
            i['Title'] = i.Name.str.extract('([A-Za-z]+)\.', expand=False)  # ???    #i에 Name에 들어있는 값을 뽑아내라
        for i in combine:
            i['Title'] = i['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            i['Title'] = i['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'],
                                            'Rare')
            i['Title'] = i['Title'].replace(['Mile', 'Mr'])
            i['Title'] = i['Title'].replace(['Ms', 'Miss'])
            i['Title'] = i['Title'].fillna(0)
            i['Title'] = i['Title'].map({
                'Mr': 1,
                'Miss': 2,
                'Mrs': 3,
                'Master': 4,
                'Royal': 5,
                'Rare': 6
            })
        return this

    @staticmethod
    def create_k_fold() -> object:
        return KFold(n_splits=10, shuffle=True, random_state=0)

    @staticmethod
    def get_accuracy(this, algo):

        score = cross_val_score(RandomForestClassifier(),
                                this.train,
                                this.label,
                                cv=Titanic_Model.create_k_fold(),
                                n_jobs=1,
                                scoring='accuracy')
        return round(np.mean(score) * 100, 2)


if __name__ == "__main__":
    t = Titanic_Model()
    this = Dataset()
    this.train = t.new_model('train.csv')
    this.test = t.new_model('train.csv')
    this = Titanic_Model.embarked_nominal(this)

"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
시각화를 통해 얻은 상관관계 변수(variable = feature = column / aspect에 따라 다른 명칭)는
pclass, 
age, 
sex, 
Fare, 
Enbarked
 === null 값 ===
 Age            177
 Cabin          687
 Embarked         2
"""


class Titanic_Controller(object):
    model = Titanic_Model()
    dataset = Dataset()

    def __init__(self):
        pass

    def __str__(self):
        return f""

    def preprocess(self, train, test) -> object:  # 모델링 / 전처리 -> 데이터에 자연어를 기계어로
        model = self.model
        this = self.dataset
        this.train = model.new_model(train)  # new_model = 프로토타입 /  train.csv 가 데이터프레임으로 전환된 객체
        this.test = model.new_model(test)  # train test 두개 합치면 this...?
        this.id = this.test['PassengerId']
        # colums
        # this = model.pclass_ordinal(this) ->이미 ordinal #this.train, this.test
        this = model.age_ordinal(this)
        this = model.sex_nominal(this)
        this = model.fare_ordinal(this)
        this = model.embarked_nominal(this)
        this = model.title_nominal(this)

        this = model.drop_features(this, 'PassengerId', 'Name', 'Sex', 'Age', 'SibSp',
                                   'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')
        return this

    def modeling(self, train, test) -> object:  # 모델링
        model = self.model
        this = self.preprocess(train, test)
        this.label = model.create_label(this)
        this.train = model.create_train(this)
        return this

    def learning(self, train, test, algo):  # 기계학습
        this = self.modeling(train, test)
        accuracy = self.model.get_accuracy(this, algo)
        print(f"랜덤포레스트분류기 정확도: {accuracy} %")

    def submit(self):  # 배포
        pass

'''
if __name__ == '__main__':
    this = Dataset()
    a = Titanic_Controller()
    b = a.preprocess("train.csv", "test.csv")
    print(f"a :{b.train.columns}\n{b.train}")
'''

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

class Plot (object):
    dataset = Dataset()
    model = Titanic_Model()

    def __init__(self, fname):
        self.entry = self.model.new_model(fname) #파일명

    def __str__(self):
        return f""

    def draw_survived(self):
        this = self.entry
        f, ax = plt.subplots(1, 2, figsize=(18, 8)) #한 화면에 두개 그래프를 그릴때는 subplots
        this['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        ax[0].set_title('0.사망자 vs 1.생존자')
        ax[0].set_ylabel('')
        ax[1].set_title('0.사망자 vs 1.생존자')
        sns.countplot(x='Survived', data=this, ax=ax[1])
        plt.show()

    def draw_pclass(self):
        this = self.entry
        this["생존결과"] = this["Survived"].replace(0, "사망자").replace(1, "생존자")
        this["좌석등급"] = this["Pclass"].replace(1, "1등석").replace(2, "2등석").replace(3, "3등석")
        sns.countplot(data=this, x="좌석등급", hue="생존결과")
        plt.show()

    def draw_sex(self):
        this = self.entry
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        this['Survived'][this['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
        this['Survived'][this['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
        ax[0].set_title('남성의 생존 비율 [0.사망자 vs 1.생존자]')
        ax[1].set_title('여성의 생존 비율 [0.사망자 vs 1.생존자]')
        plt.show()

    def draw_embarked(self):
        this = self.entry
        this["생존결과"] = this["Survived"].replace(0, "사망자").replace(1, "생존자")
        this["승선항구"] = this["Embarked"].replace("C", "쉘버그").replace("S", "사우스헴튼").replace("Q", "퀸즈타운")
        sns.countplot(data=this, x="승선항구", hue="생존결과")
        plt.show()
import numpy as np
import pandas as pd


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


from src.cmm.service.menu import Menu

if __name__ == '__main__': # 왜????
    api = Titanic_Controller()
    while 1:
        menu = Menu.print_menu(["종료", "시각화", "모델링", "머신 러닝", "배포"])
        if menu == '0':
            print('종료')
            break
        elif menu == '1':
            print('시각화')
            which = int(input("1. survived\n2. sex\n3. embarked\n4. pclass\n고르시오 : "))
            plot = Plot('train.csv') #반환값 object였으면 '' 없어야함

            if which == 1:
                plot.draw_survived()
            elif which == 2:
                plot.draw_sex()
            elif which == 3:
                plot.draw_embarked()
                #Titanic_Model.new_model(b) ????????????
            elif which == 4:
                plot.draw_pclass()
            else:
                print("잘못 입력")

        elif menu == '2':
            print('모델링')
            this = api.modeling('train.csv', 'test.csv')
            print(this.train.head())
            print(this.train.columns)

        elif menu == '3':
            print(" ### 머신러닝 ### ")
            api.learning('train.csv', 'test.csv', "랜덤포레스트분류기")
            #랜덤포레스트분류기
            #결정트리분류기 DecisionTreeClassifier
            #로지스틱회귀 LogisticRegression
            #서포트벡터머신
        elif menu == '4':
            print('배포')
            api.submit('train.csv', 'test.csv')

        else:
            print("해당 메뉴 없음")
