import tensorflow as tf
import pandas as pd
import numpy as np
# import ssl

# custom class
from config import Config

# original file columns
# customer
USER_COLUMN = 'user'
AGE_COLUMN = 'age'
GENDER_COLUMN = 'gender'
MARITAL_STATUS_COLUMN = 'marital_status'
HAVE_CHILD_COLUMN = 'have_child'
EDUCATION_COLUMN = 'education'

# product
PRODUCT_COLUMN = 'product_name'
PRODUCT_3_YR_RETURN_COLUMN = '3year_return'
PRODUCT_STD_DEV_COLUMN = 'standard_deviation'
PRODUCT_DEVIDEND_COLUMN = 'dividend'
PRODUCT_ASSET_CLASS_COLUMN = 'asset_class'

# pre-process column
# customer
AGE_CATEGORY_COLUMN = 'age_category'

# index column
# customer
USER_INDEX_COLUMN = 'user_index'
AGE_INDEX_COLUMN = 'age_index'
GENDER_INDEX_COLUMN = 'gender_index'
MARITAL_STATUS_INDEX_COLUMN = 'marital_status_index'
HAVE_CHILD_INDEX_COLUMN = 'have_child_index'
EDUCATION_INDEX_COLUMN = 'education_index'

# product
PRODUCT_INDEX_COLUMN = 'product_index'
PRODUCT_ASSET_CLASS_INDEX_COLUMN = 'asset_class_index'

IN_HEADERS = [USER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, HAVE_CHILD_COLUMN, EDUCATION_COLUMN,
              PRODUCT_COLUMN, PRODUCT_3_YR_RETURN_COLUMN, PRODUCT_STD_DEV_COLUMN, PRODUCT_DEVIDEND_COLUMN, PRODUCT_ASSET_CLASS_COLUMN]

OUT_HEADERS = [USER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, HAVE_CHILD_COLUMN, EDUCATION_COLUMN,
               PRODUCT_COLUMN, PRODUCT_3_YR_RETURN_COLUMN, PRODUCT_STD_DEV_COLUMN, PRODUCT_DEVIDEND_COLUMN, PRODUCT_ASSET_CLASS_COLUMN,
               AGE_CATEGORY_COLUMN,
               USER_INDEX_COLUMN,
               AGE_INDEX_COLUMN,
               GENDER_INDEX_COLUMN,
               EDUCATION_INDEX_COLUMN,
               HAVE_CHILD_INDEX_COLUMN,
               MARITAL_STATUS_INDEX_COLUMN,
               PRODUCT_INDEX_COLUMN,
               PRODUCT_ASSET_CLASS_INDEX_COLUMN]

PROBABILITY_COLUMN = 'probability (%)'

# read file
in_df = pd.read_csv(Config.getNNFileInput(),
                    sep=",",
                    names=IN_HEADERS,
                    header=0,
                    dtype={
    USER_COLUMN: 'str',
    AGE_COLUMN: np.int32,
    GENDER_COLUMN: 'str',
    MARITAL_STATUS_COLUMN: 'str',
    HAVE_CHILD_COLUMN: 'str',
    EDUCATION_COLUMN: 'str',
    PRODUCT_COLUMN: 'str',
    PRODUCT_3_YR_RETURN_COLUMN: np.float64,
    PRODUCT_STD_DEV_COLUMN: np.float64,
    PRODUCT_DEVIDEND_COLUMN: np.float64,
    PRODUCT_ASSET_CLASS_COLUMN: 'str'
})

# read file
# out_df = pd.read_csv(Config.getNNFileOutput(),
#                      sep=",",
#                      names=OUT_HEADERS,
#                      header=0,
#                      dtype={
#                      USER_COLUMN: 'str',
#                      AGE_COLUMN: np.int32,
#                      GENDER_COLUMN: 'str',
#                      MARITAL_STATUS_COLUMN: 'str',
#                      HAVE_CHILD_COLUMN: 'str',
#                      EDUCATION_COLUMN: 'str',
#                      PRODUCT_COLUMN: 'str',
#                      PRODUCT_3_YR_RETURN_COLUMN: np.float64,
#                      PRODUCT_STD_DEV_COLUMN: np.float64,
#                      PRODUCT_DEVIDEND_COLUMN: np.float64,
#                      PRODUCT_ASSET_CLASS_COLUMN: 'str',
#                      AGE_CATEGORY_COLUMN: 'str',
#                      USER_INDEX_COLUMN: np.int32,
#                      AGE_INDEX_COLUMN: np.int32,
#                      GENDER_INDEX_COLUMN: np.int32,
#                      EDUCATION_INDEX_COLUMN: np.int32,
#                      HAVE_CHILD_INDEX_COLUMN: np.int32,
#                      MARITAL_STATUS_INDEX_COLUMN: np.int32,
#                      PRODUCT_INDEX_COLUMN: np.int32,
#                      PRODUCT_ASSET_CLASS_INDEX_COLUMN: np.int32
#                      })


class DataStore:
    @staticmethod
    def getNNFileInput():
        return in_df


def main():
    print(DataStore.getNNFileInput())


if __name__ == '__main__':
    main()
