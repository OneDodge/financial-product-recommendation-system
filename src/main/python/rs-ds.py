
import tensorflow as tf
import pandas as pd
import numpy as np
# import ssl

FLAGS = flags.FLAGS

USER_COLUMN = 'user'
AGE_COLUMN = 'age'
GENDER_COLUMN = 'gender'
MARITAL_STATUS_COLUMN = 'marital_status'
HAVE_CHILD_COLUMN = 'have_child'
EDUCATION_COLUMN = 'education'

PRODUCT_COLUMN = 'product_name'
PRODUCT_3_YR_RETURN_COLUMN = '3year_return'
PRODUCT_STD_DEV_COLUMN = 'standard_deviation'
PRODUCT_DEVIDEND_COLUMN = 'dividend'
PRODUCT_ASSET_CLASS_COLUMN = 'asset_class'

USER_INDEX_COLUMN = 'user_index'
AGE_CATEGORY_COLUMN = 'age_category'
AGE_INDEX_COLUMN = 'age_index'
GENDER_INDEX_COLUMN = 'gender_index'
MARITAL_STATUS_INDEX_COLUMN = 'marital_status_index'
HAVE_CHILD_INDEX_COLUMN = 'have_child_index'
EDUCATION_INDEX_COLUMN = 'education_index'


PRODUCT_INDEX_COLUMN = 'product_index'
PRODUCT_ASSET_CLASS_INDEX_COLUMN = 'asset_class_index'

HEADERS = [USER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, HAVE_CHILD_COLUMN, EDUCATION_COLUMN,
           PRODUCT_COLUMN, PRODUCT_3_YR_RETURN_COLUMN, PRODUCT_STD_DEV_COLUMN, PRODUCT_DEVIDEND_COLUMN, PRODUCT_ASSET_CLASS_COLUMN]

PROBABILITY_COLUMN = 'probability (%)'


def main():
    a = 1


if __name__ == '__main__':
    main()
