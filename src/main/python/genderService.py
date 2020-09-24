
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from genderRepository import GenderRepository
import genderRepository

df = GenderRepository.readIndex()


class GenderService:
    @staticmethod
    def getGender(gender):
        if df[df[genderRepository.GENDER_COLUMN] == gender].empty:
            return df[genderRepository.GENDER_INDEX_COLUMN].max() + 1
        else:
            return df[df[genderRepository.GENDER_COLUMN] == gender][genderRepository.GENDER_INDEX_COLUMN].to_numpy()[0]


def main():
    print(GenderService.getGender('M'))
    print(GenderService.getGender('F'))


if __name__ == '__main__':
    main()
