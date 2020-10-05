import tensorflow as tf
import pandas as pd
import numpy as np
# import ssl

# custom class
from config import Config

CUSTOMER_COLUMN = 'customer'
AGE_COLUMN = 'age'
GENDER_COLUMN = 'gender'
MARITAL_STATUS_COLUMN = 'marital'
EDUCATION_LEVEL_COLUMN = 'edu_level'
NUMBER_OF_CHILD_COLUMN = 'num_of_child'
RISK_LEVEL_COLUMN = 'risk_level'
TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN = 'total_tcr'
SALARY_COLUMN = 'salary'
NCR_INDICATOR_COLUMN = 'ncr_indicator'

SECURITY_CODE_COLUMN = 'security_code'
SUB_TYPE_CODE_COLUMN = 'sub_type_code'
SECURITY_RISK_LEVEL_COLUMN = 'security_risk_level'
PRICE_CCY_COLUMN = 'price_ccy'
ASSET_CLASS_COLUMN = 'asset_class'

USER_INDEX_COLUMN = 'customer_index'
PRODUCT_INDEX_COLUMN = 'product_index'

IN_HEADERS = [CUSTOMER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, EDUCATION_LEVEL_COLUMN, NUMBER_OF_CHILD_COLUMN,
              RISK_LEVEL_COLUMN, TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN, SALARY_COLUMN, NCR_INDICATOR_COLUMN, SECURITY_CODE_COLUMN, SUB_TYPE_CODE_COLUMN,
              SECURITY_RISK_LEVEL_COLUMN, PRICE_CCY_COLUMN, ASSET_CLASS_COLUMN]

PROBABILITY_COLUMN = 'probability (%)'

# read file
in_df = pd.read_csv(Config.getNNProcessingFileInput(),
                    sep=",",
                    names=IN_HEADERS,
                    header=0,
                    dtype={
    CUSTOMER_COLUMN: 'str',
    AGE_COLUMN: np.int32,
    GENDER_COLUMN: 'str',
    MARITAL_STATUS_COLUMN: 'str',
    EDUCATION_LEVEL_COLUMN: 'str',
    NUMBER_OF_CHILD_COLUMN: np.int32,
    RISK_LEVEL_COLUMN: np.int32,
    TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN: np.float64,
    SALARY_COLUMN: np.float64,
    NCR_INDICATOR_COLUMN: 'str',
    SECURITY_CODE_COLUMN: 'str',
    SUB_TYPE_CODE_COLUMN: 'str',
    SECURITY_RISK_LEVEL_COLUMN: np.int32,
    PRICE_CCY_COLUMN: 'str',
    ASSET_CLASS_COLUMN: 'str'
})


class DataStore:
    @staticmethod
    def getNNFileInput():
        return in_df


def main():
    print(DataStore.getNNFileInput())


if __name__ == '__main__':
    main()
