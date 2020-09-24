
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

GENDER_COLUMN = 'gender'
GENDER_INDEX_COLUMN = 'gender_index'

HEADERS = [GENDER_COLUMN, GENDER_INDEX_COLUMN]


class GenderRepository:
    @staticmethod
    def generateIndex():
        names = Config.get()['genderRepository']['name'].split(",")

        df = pd.DataFrame(names)
        df.columns = [GENDER_COLUMN]

        df[GENDER_INDEX_COLUMN] = df[GENDER_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() + "gender.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "gender.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            GENDER_COLUMN: 'str',
            GENDER_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(GenderRepository.generateIndex())
    print(GenderRepository.readIndex())


if __name__ == '__main__':
    main()
