import pandas as pd

from src.utl import Dataset


class Bicycle_Model(object):

    dataset = Dataset()

    def __init__(self):
        pass

    def __str__(self):
        pass

    def preprocess(self):
        pass

    def new_model(self,fname):
        this = self.dataset
        this.context = './data/'
        this.fname = fname
        df = pd.read_csv(this.context + this.fname)
        print(df)
        return df

    def create_train(self):
        pass

    def create_label(self):
        pass
