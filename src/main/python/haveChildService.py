
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from haveChildRepository import HaveChildRepository
import haveChildRepository

df = HaveChildRepository.readIndex()


class HaveChildService:
    @staticmethod
    def getHaveChild(haveChild):
        if df[df[haveChildRepository.HAVE_CHILD_COLUMN] == haveChild].empty:
            return df[haveChildRepository.HAVE_CHILD_INDEX_COLUMN].max() + 1
        else:
            return df[df[haveChildRepository.HAVE_CHILD_COLUMN] == haveChild][haveChildRepository.HAVE_CHILD_INDEX_COLUMN].to_numpy()[0]


def main():
    print(HaveChildService.getHaveChild('Y'))
    print(HaveChildService.getHaveChild('N'))


if __name__ == '__main__':
    main()
