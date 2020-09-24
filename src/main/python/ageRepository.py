
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

AGE_COLUMN = 'age'
AGE_CATEGORY_COLUMN = 'age_category'
AGE_INDEX_COLUMN = 'age_index'

HEADERS = [AGE_COLUMN, AGE_CATEGORY_COLUMN, AGE_INDEX_COLUMN]


class AgeRepository:
    @staticmethod
    def generateIndex():
        series = np.arange(0, 150)
        df = pd.DataFrame(series)
        df.columns = [AGE_COLUMN]

        groups = Config.get()['ageRepository']['category']['groups'].split(",")
        boundries = np.array(Config.get()[
            'ageRepository']['category']['boundries'].split(",")).astype(int)
        boundries = np.append(boundries, [np.inf])

        df[AGE_CATEGORY_COLUMN] = pd.cut(
            df[AGE_COLUMN], boundries, labels=groups)

        df[AGE_INDEX_COLUMN] = df[AGE_CATEGORY_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() + "age.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "age.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            AGE_COLUMN: np.int32,
            AGE_CATEGORY_COLUMN: 'str',
            AGE_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(AgeRepository.generateIndex())
    print(AgeRepository.readIndex())


if __name__ == '__main__':
    main()
