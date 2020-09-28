
import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import HDFStore
import math
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

PRODUCT_ALTERNATIVE_CODE_COLUMN = 'product_alternative_code'


def KMBT2Number(s):
    if s != s:
        return s

    if s.find('K') > -1:
        return float(s.split('K')[0]) * 1000
    elif s.find('M') > -1:
        return float(s.split('M')[0]) * 1000000
    elif s.find('B') > -1:
        return float(s.split('B')[0]) * 1000000000000
    elif s.find('T') > -1:
        return float(s.split('T')[0]) * 1000000000000000000
    else:
        return s


def main():
    # read file
    pre_processing_file_df = pd.read_csv(Config.getNNPreProcessingCustomerFileInput(),
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

    product_file_df = pd.read_csv(Config.getNNPreProcessingProductFileInput(),
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

    product_file_df[CHANGE_PERCENTAGE_COLUMN] = product_file_df[CHANGE_PERCENTAGE_COLUMN].apply(
        lambda x: str(x).split('%')[0])

    product_file_df[MARKET_CAPTIAL_COLUMN] = product_file_df[MARKET_CAPTIAL_COLUMN].apply(
        lambda x: KMBT2Number(x))

    product_file_df[REVENUE_COLUMN] = product_file_df[REVENUE_COLUMN].apply(
        lambda x: KMBT2Number(x))

    product_file_df[VOLUME_COLUMN] = product_file_df[VOLUME_COLUMN].apply(
        lambda x: KMBT2Number(x))

    product_file_df[TOTAL_CASH_COLUMN] = product_file_df[TOTAL_CASH_COLUMN].apply(
        lambda x: KMBT2Number(x))

    product_file_df[TOTAL_DEBT_COLUMN] = product_file_df[TOTAL_DEBT_COLUMN].apply(
        lambda x: KMBT2Number(x))

    product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] = product_file_df[SYMBOL_COLUMN].apply(
        lambda x: str(x).split('.')[0] if len(str(x).split(
            '.')[0]) == 5 else '0' + str(x).split('.')[0]
    )

    pre_processing_file_df[SYMBOL_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][SYMBOL_COLUMN].to_numpy()[0])

    pre_processing_file_df[NAME_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][NAME_COLUMN].to_numpy()[0])

    pre_processing_file_df[PRICE_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][PRICE_COLUMN].to_numpy()[0])

    pre_processing_file_df[CHANGE_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][CHANGE_COLUMN].to_numpy()[0])

    pre_processing_file_df[CHANGE_PERCENTAGE_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][CHANGE_PERCENTAGE_COLUMN].to_numpy()[0])

    pre_processing_file_df[MARKET_CAPTIAL_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][MARKET_CAPTIAL_COLUMN].to_numpy()[0])

    pre_processing_file_df[TRAILING_P_E_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][TRAILING_P_E_COLUMN].to_numpy()[0])

    pre_processing_file_df[REVENUE_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][REVENUE_COLUMN].to_numpy()[0])

    pre_processing_file_df[VOLUME_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][VOLUME_COLUMN].to_numpy()[0])

    pre_processing_file_df[TOTAL_CASH_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][TOTAL_CASH_COLUMN].to_numpy()[0])

    pre_processing_file_df[TOTAL_DEBT_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][TOTAL_DEBT_COLUMN].to_numpy()[0])

    pre_processing_file_df[FIVE_YEAR_AVERAGE_DIVIDEND_YIELD] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][FIVE_YEAR_AVERAGE_DIVIDEND_YIELD].to_numpy()[0])

    pre_processing_file_df[SECTOR_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][SECTOR_COLUMN].to_numpy()[0])

    pre_processing_file_df[INDUSTRY_COLUMN] = pre_processing_file_df[SECURITY_CODE_COLUMN].apply(
        lambda x: product_file_df[product_file_df[PRODUCT_ALTERNATIVE_CODE_COLUMN] == x][INDUSTRY_COLUMN].to_numpy()[0])

    pre_processing_file_df.to_csv(
        Config.getNNPreProcessingFileOutput(), index=False)


if __name__ == '__main__':
    main()
