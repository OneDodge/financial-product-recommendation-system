
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

HAVE_CHILD_COLUMN = 'have_child'
HAVE_CHILD_INDEX_COLUMN = 'have_child_index'

HEADERS = [HAVE_CHILD_COLUMN, HAVE_CHILD_INDEX_COLUMN]


class HaveChildRepository:
    @staticmethod
    def generateIndex():
        names = Config.get()['haveChildRepository']['name'].split(",")

        df = pd.DataFrame(names)
        df.columns = [HAVE_CHILD_COLUMN]

        df[HAVE_CHILD_INDEX_COLUMN] = df[HAVE_CHILD_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() + "have_child.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "have_child.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            HAVE_CHILD_COLUMN: 'str',
            HAVE_CHILD_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(HaveChildRepository.generateIndex())
    print(HaveChildRepository.readIndex())


if __name__ == '__main__':
    main()
