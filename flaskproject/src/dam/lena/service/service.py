import copy
from io import BytesIO

import cv2
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from sphinx.util import requests


import cv2 as cv

from matplotlib import pyplot as plt

#from src.dam.lena.service import Execute, Hough, MultiMosaic, getcoordinate, \
    #showimgbyimshow
from src.cmm.service.dataset import Dataset
from src.cmm.service.menu import Menu

def LennaModels(input):
    headers = {'User-Agent': 'My User Agent 1.0'}
    res = requests.get(input, headers=headers)

    return np.array(Image.open(BytesIO(res.content)))


def canny_model(input):
    # 아래 3줄 크롤링
    headers = {'User-Agent': 'My User Agent 1.0'}
    res = requests.get(input, headers=headers)

    return np.array(Image.open(BytesIO(res.content)))


def toimg(self, fname):
    this = self.dataset
    this.context = './data/'
    this.fname = fname
    img = cv2.imread(this.context + fname, cv2.IMREAD_COLOR)
    return img


def canny(self, src):
    src = self.gaussian_filter(src)  # 케니 안에 수행 과정이 아래 4개니 일단 쓰고 봄
    src = self.gradient_calc(src)  # src = source 줄인말
    src = self.non_maximum_suppression(src)
    src = self.edge_tracking(src)


def gradient_calc(self):
    pass


def non_maximum_suppression(self):
    pass


class GaussianBlur(object):  # 두개로 쪼개짐
    '''
    def __init__(self, src, sigmax, sigmay ):
        self.src = src
        self.sigmax = sigmax
        self.sigmay = sigmay

    def get(self):
        sigmax = self.sigmax
        sigmay = self.sigmay
        src = self.src
        # 가로 커널과 세로 커널 행렬을 생성
        i = np.arange(-4 * sigmax, 4 * sigmax + 1)
        j = np.arange(-4 * sigmay, 4 * sigmay + 1)
        # 가우시안 계산
        mask = np.exp(-(i ** 2 / (2 * sigmax ** 2))) / (np.sqrt(2 * np.pi) * sigmax)
        maskT = np.exp(-(j ** 2 / (2 * sigmay ** 2))) / (np.sqrt(2 * np.pi) * sigmay)
        mask = mask[:, np.newaxis]
        maskT = maskT[:, np.newaxis].T
        return filter2D(filter2D(src, mask), maskT)  # 두번 필터링
    '''


def filter2D(src, kernel, delta=0):
    # 가장자리 픽셀을 (커널의 길이 // 2) 만큼 늘리고 새로운 행렬에 저장
    halfX = kernel.shape[0] // 2
    halfY = kernel.shape[1] // 2
    cornerPixel = np.zeros((src.shape[0] + halfX * 2, src.shape[1] + halfY * 2), dtype=np.uint8)

    # (커널의 길이 // 2) 만큼 가장자리에서 안쪽(여기서는 1만큼 안쪽)에 있는 픽셀들의 값을 입력 이미지의 값으로 바꾸어 가장자리에 0을 추가한 효과를 봄
    cornerPixel[halfX:cornerPixel.shape[0] - halfX, halfY:cornerPixel.shape[1] - halfY] = src

    dst = np.zeros((src.shape[0], src.shape[1]), dtype=np.float64)

    for y in np.arange(src.shape[1]):
        for x in np.arange(src.shape[0]):
            # 필터링 연산
            dst[x, y] = (kernel * cornerPixel[x: x + kernel.shape[0], y: y + kernel.shape[1]]).sum() + delta
    return dst


def imshow(img):
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()


def gray_scale(img):
    dst = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.229  # GRAYSCALE 변환 공식
    return dst


"""
def new_model(self,fname) ->object:
    static = cv2.imread('./data/'+fname)
    return static

def image_read(fname) ->object:
    return (lambda x: cv.imread('./data/'+x)(fname)

"""


class CannyModel(object):
    def __init__(self):
        self.img = Image.open(BytesIO(
            requests.get("https://slack-imgs.com/?c=1&o1=ro&url=https%3A%2F%2Fdocs.opencv.org%2F4.x%2Froi.jpg",
                         headers={'User-Agent': 'My User Agent 1.0'}).content))

    @staticmethod
    def imgconvertoarray(inputimg):
        return np.array(inputimg)

    @staticmethod
    def edge(img):
        return cv.Canny(np.array(img), 100, 200)

    def b(self):
        plt.subplot(121), plt.imshow(self.img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(self.edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        """
        headers = {'User-Agent': 'My User Agent 1.0'}
        self.lenna = Image.open(BytesIO(requests.get("https://app.slack.com/client/T045P4PJW1W/C046AQBH3AB", headers=headers).content))
        
         ### 디스크에서 읽는 경우 ###
            # static = cv.imread('./data/roi.jpg', 0)
            # static = cv.imread(static, 0)
            ### 메모리에서 읽는 경우 BEGIN ###
            fname = "https://docs.opencv.org/4.x/roi.jpg"
            static = Image.open(BytesIO(requests.get(fname,
                         headers={'User-Agent': 'My User Agent 1.0'}).content))
            print(f'static type : {type(static)}')
            static = np.array(static)
            ### 메모리에서 읽는 경우 END ###
        
        
        
     
   """


##################################################################
def image_read(fname) -> object:
    return (lambda x: cv.imread('./data/' + x))(fname)


def ImgToNumArray(url) -> object:
    res = requests.get(url, headers={'User-Agent': 'My User Agent 1.0'})
    image = Image.open(BytesIO(res.content))
    return np.array(image)


# headers = {'User-Agent': 'My User Agent 1.0'}

def GaussianBlur(src, sigmax, sigmay):
    # 가로 커널과 세로 커널 행렬을 생성
    i = np.arange(-4 * sigmax, 4 * sigmax + 1)
    j = np.arange(-4 * sigmay, 4 * sigmay + 1)
    # 가우시안 계산
    mask = np.exp(-(i ** 2 / (2 * sigmax ** 2))) / (np.sqrt(2 * np.pi) * sigmax)
    maskT = np.exp(-(j ** 2 / (2 * sigmay ** 2))) / (np.sqrt(2 * np.pi) * sigmay)
    mask = mask[:, np.newaxis]
    maskT = maskT[:, np.newaxis].T
    return filter2D(filter2D(src, mask), maskT)  # 두번 필터링


def Canny(src, lowThreshold, highThreshold):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # x축 소벨 행렬로 미분
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # y축 소벨 행렬로 미분
    Ix = filter2D(src, Kx)
    Iy = filter2D(src, Ky)
    G = np.hypot(Ix, Iy)  # 피타고라스 빗변 구하기
    img = G / G.max() * 255  # 엣지를 그레이스케일로 표현
    D = np.arctan2(Iy, Ix)  # 아크탄젠트 이용해서 그래디언트를 구함

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)  # 이미지 크기만큼의 행렬을 생성
    angle = D * 180. / np.pi  # 라디안을 degree로 변환(정확하지 않음)
    angle[angle < 0] += 180  # 음수일 때 180을 더함

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):  # 주변 픽셀(q, r)보다 크면 static 행렬의 값을 그대로 사용
                    Z[i, j] = img[i, j]
                else:  # 그렇지 않을 경우 0을 사용
                    Z[i, j] = 0

            except IndexError as e:  # 인덱싱 예외 발생 시 pass
                pass

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)  # 약한 에지
    strong = np.int32(255)  # 강한 에지

    # 이중 임곗값 비교

    # 최대 임곗값보다 큰 원소의 인덱스를 저장
    strong_i, strong_j = np.where(img >= highThreshold)
    # 최소 임곗값보다 작은 원소의 인덱스를 저장
    zeros_i, zeros_j = np.where(img < lowThreshold)

    # 최소 임곗값과 최대 임곗값 사이에 있는 원소의 인덱스를 저장
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    # 각각 강한 에지와 약한 에지의 값으로 저장
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):  # 강한 에지와 연결 되어있을 때
                        img[i, j] = strong  # 연결되어 있는 에지 또한 강한 에지가 됨
                    else:  # 연결되어 있지 않을 때
                        img[i, j] = 0  # 에지가 없는 0으로 설정
                except IndexError as e:
                    pass
    return img


#####################################################################################################################
ds = Dataset()


def Execute(*paras):
    cmd = paras[0]
    target = paras[1]
    if cmd == 'readimgbyfile':
        return (lambda x: cv.imread('C:/Users/gpark/PycharmProjects/flaskProject/static/data/dam/lena/data/' + x))(target)
    elif cmd == 'togray':
        return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
    elif cmd == 'toimg':
        return (lambda x: Image.fromarray(x))(target)
    elif cmd == 'readimgbyurl':
        return (lambda x: np.array(
            Image.open(BytesIO(requests.get(x, headers={'User-Agent': 'My User Agent 1.0'}).content))))(target)
    else:
        print("잘못된 입력으로 어떤 것도 실행 안됨")

######################################################################################
def Hough(edges):
    outcome = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=5)
    dst = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    if outcome is not None:
        for i in range(outcome.shape[0]):
            pt1 = (outcome[i][0][0], outcome[i][0][1])
            pt2 = (outcome[i][0][2], outcome[i][0][3])
            cv.line(dst, pt1, pt2, (255, 0, 0), 2, cv.LINE_AA)
    return dst


######################################################################################
def Haarcollection(ds, img, paras):
    coordinatels = getcoordinate(ds, img, paras)
    img2 = copy.deepcopy(img)
    img2 = drawrectangle(coordinatels, img2)
    return img2


def getcoordinate(ds, img):
    haar = cv.CascadeClassifier('C:/Users/gpark/PycharmProjects/flaskProject/static/data/dam/lena/data/haarcascade_frontalface_alt.xml')
    dimg = copy.deepcopy(img)
    face = haar.detectMultiScale(dimg, minSize=(150, 150))
    return face

def drawrectangle(face, girl):
    if len(face) == 0:
        print("인식 실패")
        quit()
    for (x, y, w, h) in face:
        print(f'얼굴 좌표 : {x},{y},{w},{h}')
        boxcolor = (255, 0, 0)  # BGR ,RGB
        cv.rectangle(girl, (x, y), (x + w, y + h), boxcolor, thickness=20)
        # BRG상태
    girl = cv.cvtColor(girl, cv.COLOR_BGR2RGB)
    # RGB상태
    cv.imwrite(f'{ds.context}girl-face.png', girl)
    girl = cv.cvtColor(girl, cv.COLOR_RGB2BGR)
    # BGR상태
    return girl
################################################################################################
def Mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    width = x2 - x1
    height = y2 - y1
    i_rect = img[y1:y2, x1:x2]
    i_small = cv.resize(i_rect, (size, size))
    i_mos = cv.resize(i_small, (width, height), interpolation=cv.INTER_AREA)
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2

def MultiMosaic(coordinatels, inputimg):
    for (x, y, w, h) in coordinatels:
        face_regulator = [x, y, x + w, y + h]
        inputimg = Mosaic(inputimg, face_regulator, 10)
    return inputimg
#####################################################################
def showimgbyimshow(arrimg):
    cv.imshow('showimg', arrimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

def showimgbyplt(arrimg):
    plt.imshow(arrimg, cmap='gray')
    plt.show()

def showimgprogressbyplt(*imgprogress):
    j=1
    for i in imgprogress:
        plt.subplot(1,len(imgprogress),j), plt.imshow(i, cmap='gray')
        plt.title(f'{i}'), plt.xticks([]), plt.yticks([])
        j = j + 1
        plt.show()

submenuls = ['이미지 파일 불러오기', 'URL로 이미지 불러오기']
class MenuController(object):
    @staticmethod
    def menu_0(*paras):   #종료
        print(paras[0])

    @staticmethod
    def menu_1(*paras):   #원본보기
        print(paras[0])
        submenu = Menu.print_menu(submenuls)
        if submenu == '0':
            arrimg = Execute('readimgbyfile', input('파일 이름 : '))

            showimgbyimshow(arrimg)

        elif submenu == '1':
            arrimg = Execute('readimgbyurl', input('URL : '))

            showimgbyimshow(arrimg)

        else:
            print('잘못 입력')

    @staticmethod
    def menu_2(*paras):   #그레이스케일
        print(paras[0])
        submenu = Menu.print_menu(submenuls)
        if submenu == '0':
            arrimg = Execute('readimgbyfile', input('파일 이름 : '))

            grayimg = Execute('togray', arrimg)

            plt.imshow(grayimg, cmap='gray')
            plt.show()

        elif submenu == '1':
            arrimg = Execute('readimgbyurl', input('URL : '))

            grayimg = Execute('togray', arrimg)

            plt.imshow(grayimg, cmap='gray')
            plt.show()

        else:
            print('잘못 입력')


    @staticmethod
    def menu_3(*paras):   #엣지검출
        print(paras[0])
        submenu = Menu.print_menu(submenuls)
        if submenu == '0':
            arrimg = Execute('readimgbyfile', input('파일 이름 : '))

            edges = cv.Canny((arrimg), 0, 100)

            plt.imshow(edges, cmap='gray')
            plt.show()

        elif submenu == '1':
            arrimg = Execute('readimgbyurl', input('URL : '))

            edges = cv.Canny((arrimg), 0, 100)

            plt.imshow(edges, cmap='gray')
            plt.show()

        else:
            print('잘못 입력')

    @staticmethod
    def menu_4(*paras):  # 직선검출
        print(paras[0])
        submenu = Menu.print_menu(submenuls)
        if submenu == '0':
            arrimg = Execute('readimgbyfile', input('파일 이름 : '))

            edges = cv.Canny((arrimg), 0, 100)

            dst = Hough(edges)
            showimgbyimshow(dst)
            '''
            plt.subplot(121), plt.imshow(arrimg, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dst, cmap='gray')
            plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
            plt.show()
            '''
        elif submenu == '1':
            arrimg = Execute('readimgbyurl', input('URL : '))

            edges = cv.Canny((arrimg), 0, 100)

            dst = Hough(edges)

            plt.subplot(121), plt.imshow(arrimg, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dst, cmap='gray')
            plt.title('Hough Image'), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            print('잘못 입력')
    @staticmethod
    def menu_5(*paras):      #모자이크
        ds = Dataset()
        print(paras[0])
        submenu = Menu.print_menu(submenuls)
        if submenu == '0':
            fname = input('파일 이름 : ')
            arrimg = Execute('readimgbyfile', fname)

            face = getcoordinate(ds, arrimg)

            pplsimg = MultiMosaic(face, arrimg)  # x,y,w,h     (x, y), (x + w, y + h)

            showimgbyimshow(pplsimg)

        elif submenu == '1':     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            URL = input('URL : ')
            arrimg = Execute('readimgbyurl', URL)

            photoname = input("사진 이름 : ")

            cv.imwrite(f'{ds.context}{photoname}', arrimg)

            face = getcoordinate(ds, arrimg)

            pplsimg = MultiMosaic(face, arrimg)  # x,y,w,h     (x, y), (x + w, y + h)

            showimgbyimshow(pplsimg)

        else:
            print('잘못입력')

GIRL = 'girl.jpg'
GIRL_INCLIEND = 'girl_incliend.png'
GIRL_SIDE_FACE = 'girl_side_face.jpg'
GIRL_WITH_MOM = 'girl_with_mom.jpg'
CAT = 'cat.jpg'
HAAR = "haarcascade_frontalface_alt.xml"

if __name__ == "__main__":
    api = MenuController()
    while 1:
        menuls = ['종료', '원본 보기', '그레이 스케일', '엣지 검출', '직선 검출', '모자이크']
        # "종료","원본보기","그레이스케일","엣지검출","직선검출","모자이크", "소녀 모자이크","모녀 모자이크"
        menu = Menu.print_menu(menuls)
        print('#' * 40)
        if menu == '0':
            api.menu_0(menuls[0])
            break
        elif menu == '1':
            api.menu_1(menuls[1])

        elif menu == '2':
            api.menu_2(menuls[2])

        elif menu == '3':
            api.menu_3(menuls[3])

        elif menu == '4':
            api.menu_4(menuls[4])

        elif menu == '5':
            api.menu_5(menuls[5])

        else:
            print("잘못 입력")


