
import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import HDFStore

import os
# import ssl

# custom class
from config import Config

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Pre-Processing File Columns
CUSTOMER_COLUMN = 'customer'
AGE_COLUMN = 'age'
GENDER_COLUMN = 'gender'
MARITAL_STATUS_COLUMN = 'marital'
EDUCATION_LEVEL_COLUMN = 'edu_level'
NUMBER_OF_CHILD_COLUMN = 'num_of_child'
RISK_LEVEL_COLUMN = 'risk_level'
TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN = 'total_tcr'
SALARY_COLUMN = 'salary'
SECURITY_CODE_COLUMN = 'security_code'

PRE_PROCESSING_FILE_HEADERS = [CUSTOMER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, EDUCATION_LEVEL_COLUMN, NUMBER_OF_CHILD_COLUMN,
                               RISK_LEVEL_COLUMN, TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN, SALARY_COLUMN, SECURITY_CODE_COLUMN]

# Product File Columns
SYMBOL_COLUMN = 'symbol'
NAME_COLUMN = 'name'
PRICE_COLUMN = 'price'
CHANGE_COLUMN = 'change'
CHANGE_PERCENTAGE_COLUMN = 'change_percentage'
MARKET_CAPTIAL_COLUMN = 'market_captial'
TRAILING_P_E_COLUMN = 'trailing_p_e'
REVENUE_COLUMN = 'revenue'
VOLUME_COLUMN = 'volume'
TOTAL_CASH_COLUMN = 'total_cash'
TOTAL_DEBT_COLUMN = 'total_debt_column'
FIVE_YEAR_AVERAGE_DIVIDEND_YIELD = '5_year_average_dividend_yield'
SECTOR_COLUMN = 'sector'
INDUSTRY_COLUMN = 'industry'

PRODUCT_FILE_HEADERS = [SYMBOL_COLUMN, NAME_COLUMN, PRICE_COLUMN, CHANGE_COLUMN, CHANGE_PERCENTAGE_COLUMN, MARKET_CAPTIAL_COLUMN,
                        TRAILING_P_E_COLUMN, REVENUE_COLUMN, VOLUME_COLUMN, TOTAL_CASH_COLUMN, TOTAL_DEBT_COLUMN,
                        FIVE_YEAR_AVERAGE_DIVIDEND_YIELD, SECTOR_COLUMN, INDUSTRY_COLUMN]


def main():
    # read file
    pre_processing_file_df = pd.read_csv(Config.getNNPreProcessingFileInput(),
                                         sep=",",
                                         names=PRE_PROCESSING_FILE_HEADERS,
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
        SECURITY_CODE_COLUMN: 'str'
    })
    print(pre_processing_file_df)

    product_file_df = pd.read_csv(Config.getNNProductFileInput(),
                                  sep=",",
                                  names=PRODUCT_FILE_HEADERS,
                                  header=0,
                                  dtype={
        SYMBOL_COLUMN: 'str',
        NAME_COLUMN: 'str',
        PRICE_COLUMN: np.float64,
        CHANGE_COLUMN: np.float64,
        CHANGE_PERCENTAGE_COLUMN: 'str',
        MARKET_CAPTIAL_COLUMN: 'str',
        TRAILING_P_E_COLUMN: 'str',
        REVENUE_COLUMN: 'str',
        VOLUME_COLUMN: 'str',
        TOTAL_CASH_COLUMN: 'str',
        TOTAL_DEBT_COLUMN: 'str',
        FIVE_YEAR_AVERAGE_DIVIDEND_YIELD: np.float64,
        SECTOR_COLUMN: 'str',
        INDUSTRY_COLUMN: 'str'
    })
    print(product_file_df)


if __name__ == '__main__':
    main()
