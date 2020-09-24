
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config

EDUCATION_COLUMN = 'education'
EDUCATION_INDEX_COLUMN = 'education_index'

HEADERS = [EDUCATION_COLUMN, EDUCATION_INDEX_COLUMN]


class EducationRepository:
    @staticmethod
    def generateIndex():
        names = Config.get()['educationRepository']['name'].split(",")

        df = pd.DataFrame(names)
        df.columns = [EDUCATION_COLUMN]

        df[EDUCATION_INDEX_COLUMN] = df[EDUCATION_COLUMN].astype(
            'category').cat.codes

        df.to_csv(Config.getNNIndex() + "education.index.csv", index=False)
        return df

    @staticmethod
    def readIndex():
        df = pd.read_csv(Config.getNNIndex() + "education.index.csv",
                         sep=",",
                         names=HEADERS,
                         header=0,
                         dtype={
            EDUCATION_COLUMN: 'str',
            EDUCATION_INDEX_COLUMN: np.int32
        })
        return df


def main():
    print(EducationRepository.generateIndex())
    print(EducationRepository.readIndex())


if __name__ == '__main__':
    main()
