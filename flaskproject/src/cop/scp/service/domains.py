# 여기서 데이터 가져오기 / 크롤링 한번하고 버릴꺼면 함수형으로, 한번하고 저장할꺼면 지금처럼/static으로 걸릴게 없음

"""
지원하는 Parser 종류
"html.parser" : 빠르지만 유연하지 않기 때문에 단순한 HTML문서에 사용합니다.
"lxml" : 매우 빠르고 유연합니다.
"xml" : XML 파일에만 사용합니다.
"html5lib" : 복잡한 구조의 HTML에 대해서 사용합니다.
"""
from dataclasses import dataclass
import pandas as pd
from bs4 import BeautifulSoup

#데이터 가공
from urllib.request import urlopen
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen




def BugsMusic(arg):
    soup = BeautifulSoup(urlopen(arg.domain + arg.query_string), 'lxml')
    title = {"class": arg.class_names[0]}  # class => 해쉬태그
    artist = {"class": arg.class_names[1]}
    titles = soup.find_all(name=arg.tag_name, attrs=title)
    titles = [i.find('a').text for i in titles]
    artists = soup.find_all(name=arg.tag_name, attrs=artist)
    artists = [i.find('a').text for i in artists]

    '''
    j = 0
    for i in titles:
        print(f"Title : {i.find('a').text} // Artist : {artists[j].find('a').text}")
        j = j + 1

    print('*'*200)
    '''
    # 디버깅
    print(titles)
    print(artists)
    #[print(f"Rank {j + 1} | Title : {i[0]} / Artist : {i[1]}") for j, i in enumerate(list(zip(titles, artists)))]
    # dict 전환
    diction = {}
    for i, j in enumerate(titles):
        diction[j] = artists[i]
    # csv로 저장
    arg.diction = diction
    arg.dict_to_dataframe()
    arg.dataframe_to_csv()

def Melon(arg):
    urlheader = urllib.request.Request(arg.domain + arg.query_string, headers={'User-Agent': "Mozilla/5.0"})
    htmlurl = urllib.request.urlopen(urlheader).read()

    soup = BeautifulSoup(htmlurl, 'lxml')
    title = {"class": arg.class_names[0]}  # class => 해쉬태그
    artist = {"class": arg.class_names[1]}
    titles = soup.find_all(name=arg.tag_name, attrs=title)
    titles = [i.find('a').text for i in titles]
    artists = soup.find_all(name=arg.tag_name, attrs=artist)
    artists = [i.find('a').text for i in artists]

    '''
    j = 0
    for i in titles:
        print(f"Title : {i.find('a').text} // Artist : {artists[j].find('a').text}")
        j = j + 1

    print('*'*200)
    '''
    # 디버깅
    [print(f"Rank {j + 1} | Title : {i[0]} / Artist : {i[1]}") for j, i in enumerate(list(zip(titles, artists)))]
    # dict 전환
    diction = {}
    for i, j in enumerate(titles):
        diction[j] = artists[i]
    # csv로 저장
    arg.diction = diction
    arg.dict_to_dataframe()
    arg.dataframe_to_csv()

@dataclass
class Scrap(object):
    #assingment
    parser = ''
    html = ''
    domain = ''
    query_string = ''
    headers = {}
    tag_name = ''
    fname = ''
    class_names = []
    artists = []
    titles = []
    diction = {}
    df = None
    soup = BeautifulSoup

    # declaration
    '''
    parser: str       
    html: str
    domain: str
    query_string: str
    headers: dict
    tag_name: str
    fname: str
    class_names: list
    artists: list
    titles: list
    diction: dict
    df: None
    soup: BeautifulSoup

    @property
    def parser(self) -> str: return self._parser
    @parser.setter
    def parser(self, parser):  self._parser = parser

    @property
    def html(self) -> str: return self._html
    @html.setter
    def html(self, html):  self._html = html

    @property
    def soup(self): return self._soup
    @soup.setter
    def soup(self, soup):  self._soup = soup

    @property
    def domain(self) -> str: return self._domain
    @domain.setter
    def domain(self, domain):  self._domain = domain

    @property
    def query_string(self) -> str: return self._query_string
    @query_string.setter
    def query_string(self, query_string):  self._query_string = query_string

    @property
    def headers(self) -> dict: return self._headers
    @headers.setter
    def headers(self, headers):  self._headers = headers

    @property
    def tag_name(self) -> str: return self._tag_name
    @tag_name.setter
    def tag_name(self, tag_name):  self._tag_name = tag_name

    @property
    def fname(self): return self._fname
    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def class_names(self): return self._class_names
    @class_names.setter
    def class_names(self, class_names):  self._class_names = class_names

    @property
    def artists(self): return self._artists
    @artists.setter
    def artists(self, artists):  self._artists = artists

    @property
    def titles(self): return self._titles
    @titles.setter
    def titles(self, titles):  self._titles = titles

    @property
    def diction(self): return self._diction
    @diction.setter
    def diction(self, diction):  self._diction = diction

    @property
    def df(self): return self._df
    @df.setter
    def df(self, df):  self._df = df

    '''

    def dict_to_dataframe(self):
        self.df = pd.DataFrame.from_dict(self.diction, orient='index')  # orient = 'index' -> index 자동 생성
    def dataframe_to_csv(self):
        path = 'C:/Users/gpark/PycharmProjects/flaskProject/static/data/cop/scp/service/save/'+self.fname+'.csv'
        self.df.to_csv(path, sep=',', na_rep='NaN')

class ScrapController(object):
    @staticmethod
    def menu0(*params):
        print(params[0])

    @staticmethod
    def menu1(*params):
        print(params[0])
        BugsMusic(params[1])

    @staticmethod
    def menu2(*params):
        print(params[0])
        Melon(params[1])

from src.cop.scp.service.domains import Scrap

from src.cmm.service.menu import Menu


if __name__ == "__main__":
    menuls = ['종료', '벅스 뮤직', '멜론']
    scrap = Scrap()
    Menu.starprinter()
    api = ScrapController()
    while 1:
        menu = Menu.print_menu(menuls)
        Menu.starprinter()
        if menu == '0':
            api.menu0(menuls[0])
            Menu.starprinter()
            break

        elif menu == '1':
            scrap.domain = 'https://music.bugs.co.kr/chart/track/day/total?chartdate='
            scrap.query_string = '20221101'
            scrap.parser = 'lxml'
            scrap.class_names = ["title", "artist"]
            scrap.tag_name = 'p'
            scrap.fname = 'bugs_result'
            api.menu1(menuls[1], scrap)
            Menu.starprinter()


        elif menu == '2':
            scrap.domain = 'https://www.melon.com/chart/index.htm?dayTime='
            scrap.query_string = '2022110808'
            scrap.parser = 'lxml'
            scrap.class_names = ["ellipsis rank01", "ellipsis rank02"]
            scrap.tag_name = 'div'
            scrap.fname = 'melon_result'
            api.menu2(menuls[2], scrap)
            Menu.starprinter()

        else:
            print('잘못 입력')
