import tensorflow as tf
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras import Model
from tensorflow.keras.models import load_model

# custom class
from config import Config
import rs_ds as ds
from rs_ds import DataStore

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.keras.backend.set_floatx('float64')

app = Flask(__name__)
CORS(app)

model = load_model(Config.getNNModel())

# read file
df = DataStore.getNNFileInput()


@app.route("/recommendation/product", methods=['GET', 'POST'])
def productRecommendationApi():
    content = request.json

    desired_user = content["customer"]
    desired_age = int(content["age"])
    desired_gender = content["gender"]
    desired_marital = content["marital"]
    desired_edu_level = content["edu_level"]
    desired_num_of_child = int(content["num_of_child"])
    desired_risk_level = int(content["risk_level"])
    desired_total_tcr = float(content["total_tcr"])
    desired_salary = float(content["salary"])
    desired_ncr_indicator = content["ncr_indicator"]

    new_table = []
    result_table = []
    for symbol in df[ds.SECURITY_CODE_COLUMN].unique():
        product_df = df[df[ds.SECURITY_CODE_COLUMN] == symbol].iloc[[0]]

        new_row = []
        new_row.append(desired_age)
        new_row.append(desired_gender)
        new_row.append(desired_marital)
        new_row.append(desired_edu_level)
        new_row.append(desired_num_of_child)
        new_row.append(desired_risk_level)
        new_row.append(desired_total_tcr)
        new_row.append(desired_salary)
        new_row.append(desired_ncr_indicator)

        new_row.append(product_df[ds.SECURITY_CODE_COLUMN].to_numpy()[0])
        new_row.append(product_df[ds.SUB_TYPE_CODE_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.SECURITY_RISK_LEVEL_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.PRICE_CCY_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.ASSET_CLASS_COLUMN].to_numpy()[
            0])

        new_table.append(new_row)
        result_item = []
        result_item.append(product_df[ds.SECURITY_CODE_COLUMN].to_numpy()[
            0])
        result_table.append(result_item)

    new_table_df = pd.DataFrame(data=np.array(new_table))
    new_table_df.columns = [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.EDUCATION_LEVEL_COLUMN, ds.NUMBER_OF_CHILD_COLUMN,
                            ds.RISK_LEVEL_COLUMN, ds.TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN, ds.SALARY_COLUMN, ds.NCR_INDICATOR_COLUMN, ds.SECURITY_CODE_COLUMN, ds.SUB_TYPE_CODE_COLUMN,
                            ds.SECURITY_RISK_LEVEL_COLUMN, ds.PRICE_CCY_COLUMN, ds.ASSET_CLASS_COLUMN]

    new_table_df[ds.AGE_COLUMN] = new_table_df[ds.AGE_COLUMN].astype(
        str).astype(int)
    new_table_df[ds.NUMBER_OF_CHILD_COLUMN] = new_table_df[ds.NUMBER_OF_CHILD_COLUMN].astype(
        str).astype(int)
    new_table_df[ds.RISK_LEVEL_COLUMN] = new_table_df[ds.RISK_LEVEL_COLUMN].astype(
        str).astype(int)
    new_table_df[ds.TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN] = new_table_df[ds.TOTAL_TOTAL_CAPTIAL_RATIO_COLUMN].astype(
        str).astype(float)
    new_table_df[ds.SALARY_COLUMN] = new_table_df[ds.SALARY_COLUMN].astype(
        str).astype(float)

    new_table_df[ds.SECURITY_RISK_LEVEL_COLUMN] = new_table_df[ds.SECURITY_RISK_LEVEL_COLUMN].astype(
        str).astype(int)

    input_dict = {col: tf.convert_to_tensor(
        new_table_df[col].to_numpy()) for col in new_table_df.columns}

    predictions = model.predict(input_dict)

    print(len(predictions))
    for i in range(len(predictions)):
        result_item = result_table[i]
        result_item.append(predictions[i][0] * 100)
        result_table[i] = result_item
    result_table = np.array(result_table)
    result_df = pd.DataFrame(data=result_table)
    result_df.columns = [ds.SECURITY_CODE_COLUMN, ds.PROBABILITY_COLUMN]
    result_df = result_df.sort_values(
        by=[ds.PROBABILITY_COLUMN], ascending=False)
    result_df[ds.PROBABILITY_COLUMN] = result_df[ds.PROBABILITY_COLUMN].astype(
        str).astype(float)
    return result_df.to_json(orient="records")


# @app.route("/recommendation/user", methods=['GET', 'POST'])
# def userRecommendationApi():
#     content = request.json

#     desired_symbol = content["symbol"]
#     desired_name = content["name"]
#     desired_price = float(content["price"])
#     desired_change = float(content["change"])
#     desired_change_percentage = float(content["change_percentage"])
#     desired_market_capital = float(content["market_captial"])
#     desired_trailing_p_e = float(content["trailing_p_e"])
#     try:
#         desired_revenue = float(content["revenue"])
#     except:
#         desired_revenue = float("nan")

#     desired_volume = float(content["volume"])

#     try:
#         desired_total_cash = float(content["total_cash"])
#     except:
#         desired_total_cash = float("nan")

#     try:
#         desired_total_debt = float(content["total_debt"])
#     except:
#         desired_total_debt = float("nan")

#     desired_5_year_average_dividend_yield = float(
#         content["5_year_average_dividend_yield"])
#     desired_sector = content["sector"]
#     desired_industry = content["industry"]

#     new_table = []
#     result_table = []
#     for user_name in df[ds.USER_COLUMN].unique():
#         user_df = df[df[ds.USER_COLUMN] == user_name].iloc[[0]]

#         new_row = []
#         new_row.append(user_df[ds.AGE_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.GENDER_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.MARITAL_STATUS_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.EDUCATION_LEVEL_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.NUMBER_OF_CHILD_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.RISK_LEVEL_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.TOTAL_TCR_COLUMN].to_numpy()[
#             0])
#         new_row.append(user_df[ds.SALARY_COLUMN].to_numpy()[
#             0])

#         new_row.append(desired_price)
#         new_row.append(desired_change)
#         new_row.append(desired_change_percentage)
#         new_row.append(desired_market_capital)
#         # new_row.append(desired_trailing_p_e)
#         # new_row.append(desired_revenue)
#         new_row.append(desired_volume)
#         # new_row.append(desired_total_cash)
#         # new_row.append(desired_total_debt)
#         new_row.append(desired_5_year_average_dividend_yield)
#         new_row.append(desired_sector)
#         new_row.append(desired_industry)

#         new_table.append(new_row)
#         result_item = []
#         result_item.append(user_df[ds.USER_COLUMN].to_numpy()[
#             0])
#         result_table.append(result_item)

#     new_table_df = pd.DataFrame(data=np.array(new_table))
#     new_table_df.columns = [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.EDUCATION_LEVEL_COLUMN, ds.NUMBER_OF_CHILD_COLUMN,
#                             ds.RISK_LEVEL_COLUMN, ds.TOTAL_TCR_COLUMN, ds.SALARY_COLUMN, ds.PRICE_COLUMN, ds.CHANGE_COLUMN,
#                             ds.CHANGE_PERCENTAGE_COLUMN, ds.MARKET_CAPTIAL_COLUMN,
#                             # ds.TRAILING_P_E_COLUMN,
#                             # ds.REVENUE_COLUMN,
#                             ds.VOLUME_COLUMN,
#                             # ds.TOTAL_CASH_COLUMN,
#                             # ds.TOTAL_DEBT_COLUMN,
#                             ds.FIVE_YEAR_AVERAGE_DIVIDEND_YIELD, ds.SECTOR_COLUMN, ds.INDUSTRY_COLUMN]

#     new_table_df[ds.AGE_COLUMN] = new_table_df[ds.AGE_COLUMN].astype(
#         str).astype(int)
#     new_table_df[ds.NUMBER_OF_CHILD_COLUMN] = new_table_df[ds.NUMBER_OF_CHILD_COLUMN].astype(
#         str).astype(int)
#     new_table_df[ds.RISK_LEVEL_COLUMN] = new_table_df[ds.RISK_LEVEL_COLUMN].astype(
#         str).astype(int)
#     new_table_df[ds.TOTAL_TCR_COLUMN] = new_table_df[ds.TOTAL_TCR_COLUMN].astype(
#         str).astype(float)
#     new_table_df[ds.SALARY_COLUMN] = new_table_df[ds.SALARY_COLUMN].astype(
#         str).astype(float)

#     new_table_df[ds.PRICE_COLUMN] = new_table_df[ds.PRICE_COLUMN].astype(
#         str).astype(float)

#     new_table_df[ds.PRICE_COLUMN] = new_table_df[ds.PRICE_COLUMN].astype(
#         str).astype(float)
#     new_table_df[ds.CHANGE_COLUMN] = new_table_df[ds.CHANGE_COLUMN].astype(
#         str).astype(float)
#     new_table_df[ds.CHANGE_PERCENTAGE_COLUMN] = new_table_df[ds.CHANGE_PERCENTAGE_COLUMN].astype(
#         str).astype(float)
#     new_table_df[ds.MARKET_CAPTIAL_COLUMN] = new_table_df[ds.MARKET_CAPTIAL_COLUMN].astype(
#         str).astype(float)
#     # new_table_df[ds.TRAILING_P_E_COLUMN] = df[ds.TRAILING_P_E_COLUMN].astype(
#     #     str).astype(float)
#     # new_table_df[ds.REVENUE_COLUMN] = new_table_df[ds.REVENUE_COLUMN].astype(
#     #     str).astype(float)
#     new_table_df[ds.VOLUME_COLUMN] = new_table_df[ds.VOLUME_COLUMN].astype(
#         str).astype(float)

#     # new_table_df[ds.TOTAL_CASH_COLUMN] = new_table_df[ds.TOTAL_CASH_COLUMN].astype(
#     #     str).astype(float)
#     # new_table_df[ds.TOTAL_DEBT_COLUMN] = new_table_df[ds.TOTAL_DEBT_COLUMN].astype(
#     #     str).astype(float)
#     new_table_df[ds.FIVE_YEAR_AVERAGE_DIVIDEND_YIELD] = new_table_df[ds.FIVE_YEAR_AVERAGE_DIVIDEND_YIELD].astype(
#         str).astype(float)

#     input_dict = {col: tf.convert_to_tensor(
#         new_table_df[col].to_numpy()) for col in new_table_df.columns}

#     predictions = model.predict(input_dict)

#     print(len(predictions))
#     for i in range(len(predictions)):
#         result_item = result_table[i]
#         result_item.append(predictions[i][0] * 100)
#         result_table[i] = result_item
#     result_table = np.array(result_table)
#     result_df = pd.DataFrame(data=result_table)
#     result_df.columns = [ds.USER_COLUMN, ds.PROBABILITY_COLUMN]
#     result_df = result_df.sort_values(
#         by=[ds.PROBABILITY_COLUMN], ascending=False)
#     result_df[ds.PROBABILITY_COLUMN] = result_df[ds.PROBABILITY_COLUMN].astype(
#         str).astype(float)
#     return result_df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
