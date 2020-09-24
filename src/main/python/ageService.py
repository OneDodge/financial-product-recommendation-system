
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from ageRepository import AgeRepository
import ageRepository

df = AgeRepository.readIndex()


class AgeService:
    @staticmethod
    def getAge(age):
        if Config.get()['ageRepository']['category']['enable'] == True:
            if df[df[ageRepository.AGE_COLUMN] == age].empty:
                return df[ageRepository.AGE_INDEX_COLUMN].max() + 1
            else:
                return df[df[ageRepository.AGE_COLUMN] == age][ageRepository.AGE_INDEX_COLUMN].to_numpy()[0]
        else:
            return age


def main():
    print(AgeService.getAge(140))
    print(AgeService.getAge(160))


if __name__ == '__main__':
    main()
