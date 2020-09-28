import tensorflow as tf
import pandas as pd
import numpy as np
# import ssl

# custom class
from config import Config

# Pre-Processing File Columns
USER_COLUMN = 'customer'
AGE_COLUMN = 'age'
GENDER_COLUMN = 'gender'
MARITAL_STATUS_COLUMN = 'marital'
EDUCATION_LEVEL_COLUMN = 'edu_level'
NUMBER_OF_CHILD_COLUMN = 'num_of_child'
RISK_LEVEL_COLUMN = 'risk_level'
TOTAL_TCR_COLUMN = 'total_tcr'
SALARY_COLUMN = 'salary'
SECURITY_CODE_COLUMN = 'security_code'
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

USER_INDEX_COLUMN = 'customer_index'
PRODUCT_INDEX_COLUMN = 'product_index'

IN_HEADERS = [USER_COLUMN, AGE_COLUMN, GENDER_COLUMN, MARITAL_STATUS_COLUMN, EDUCATION_LEVEL_COLUMN, NUMBER_OF_CHILD_COLUMN,
              RISK_LEVEL_COLUMN, TOTAL_TCR_COLUMN, SALARY_COLUMN, SECURITY_CODE_COLUMN, SYMBOL_COLUMN, NAME_COLUMN, PRICE_COLUMN, CHANGE_COLUMN,
              CHANGE_PERCENTAGE_COLUMN, MARKET_CAPTIAL_COLUMN, TRAILING_P_E_COLUMN, REVENUE_COLUMN, VOLUME_COLUMN, TOTAL_CASH_COLUMN, TOTAL_DEBT_COLUMN,
              FIVE_YEAR_AVERAGE_DIVIDEND_YIELD, SECTOR_COLUMN, INDUSTRY_COLUMN]

PROBABILITY_COLUMN = 'probability (%)'

# read file
in_df = pd.read_csv(Config.getNNProcessingFileInput(),
                    sep=",",
                    names=IN_HEADERS,
                    header=0,
                    dtype={
    USER_COLUMN: 'str',
    AGE_COLUMN: np.int32,
    GENDER_COLUMN: 'str',
    MARITAL_STATUS_COLUMN: 'str',
    EDUCATION_LEVEL_COLUMN: 'str',
    NUMBER_OF_CHILD_COLUMN: np.int32,
    RISK_LEVEL_COLUMN: np.int32,
    TOTAL_TCR_COLUMN: np.float64,
    SALARY_COLUMN: np.float64,
    SECURITY_CODE_COLUMN: 'str',
    SYMBOL_COLUMN: 'str',
    NAME_COLUMN: 'str',
    PRICE_COLUMN: np.float64,
    CHANGE_COLUMN: np.float64,
    CHANGE_PERCENTAGE_COLUMN: np.float64,
    MARKET_CAPTIAL_COLUMN: np.float64,
    TRAILING_P_E_COLUMN: np.float64,
    REVENUE_COLUMN: np.float64,
    VOLUME_COLUMN: np.float64,
    TOTAL_CASH_COLUMN: np.float64,
    TOTAL_DEBT_COLUMN: np.float64,
    FIVE_YEAR_AVERAGE_DIVIDEND_YIELD: np.float64,
    SECTOR_COLUMN: 'str',
    INDUSTRY_COLUMN: 'str'
})


class DataStore:
    @staticmethod
    def getNNFileInput():
        return in_df


def main():
    print(DataStore.getNNFileInput())


if __name__ == '__main__':
    main()
