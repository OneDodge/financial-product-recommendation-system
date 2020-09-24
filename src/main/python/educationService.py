
import numpy as np
import pandas as pd
import tensorflow as tf

# import ssl
from config import Config
from educationRepository import EducationRepository
import educationRepository

df = EducationRepository.readIndex()


class EducationService:
    @staticmethod
    def getEducation(education):
        if df[df[educationRepository.EDUCATION_COLUMN] == education].empty:
            return df[educationRepository.EDUCATION_INDEX_COLUMN].max() + 1
        else:
            return df[df[educationRepository.EDUCATION_COLUMN] == education][educationRepository.EDUCATION_INDEX_COLUMN].to_numpy()[0]


def main():
    print(EducationService.getEducation('PRIMARY'))
    print(EducationService.getEducation('POSTGRADUATE'))


if __name__ == '__main__':
    main()
