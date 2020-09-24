
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from maritalStatusRepository import MaritalStatusRepository
import maritalStatusRepository

df = MaritalStatusRepository.readIndex()


class MaritalStatusService:
    @staticmethod
    def getMaritalStatus(maritalStatus):
        if df[df[maritalStatusRepository.MARITAL_STATUS_COLUMN] == maritalStatus].empty:
            return df[maritalStatusRepository.MARITAL_STATUS_INDEX_COLUMN].max() + 1
        else:
            return df[df[maritalStatusRepository.MARITAL_STATUS_COLUMN] == maritalStatus][maritalStatusRepository.MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0]


def main():
    print(MaritalStatusService.getMaritalStatus('SINGLE'))
    print(MaritalStatusService.getMaritalStatus('MARRIED'))


if __name__ == '__main__':
    main()
