
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

MARITAL_STATUS_COLUMN = 'marital_status'
MARITAL_STATUS_INDEX_COLUMN = 'marital_status_index'

HEADERS = [MARITAL_STATUS_COLUMN, MARITAL_STATUS_INDEX_COLUMN]


class MaritalStatusRepository:
    @staticmethod
    def generateIndex():
        names = Config.get()['maritalStatusRepository']['name'].split(",")

        df = pd.DataFrame(names)
        df.columns = [MARITAL_STATUS_COLUMN]

        df[MARITAL_STATUS_INDEX_COLUMN] = df[MARITAL_STATUS_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() +
                  "marital_status.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "marital_status.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            MARITAL_STATUS_COLUMN: 'str',
            MARITAL_STATUS_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(MaritalStatusRepository.generateIndex())
    print(MaritalStatusRepository.readIndex())


if __name__ == '__main__':
    main()
