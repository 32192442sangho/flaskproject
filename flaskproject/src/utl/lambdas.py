from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from sphinx.util import requests

from src.cmm.service.dataset import Dataset

ds = Dataset()

def mosaiclambdas(*paras):
    cmd = paras[0]
    target = paras[1]
    if cmd == 'readimg':
        return (lambda x: cv2.imread(ds.context + x))(target)
    elif cmd == 'togray':
        return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
    elif cmd == 'toimg':
        return (lambda x: Image.fromarray(x))(target)
    elif cmd == 'toarray_file=url':
        return (lambda x: np.array(
            Image.open(BytesIO(requests.get(x, headers={'User-Agent': 'My User Agent 1.0'}).content))))(target)
    else:
        print("잘못된 입력으로 어떤 것도 실행 안됨")