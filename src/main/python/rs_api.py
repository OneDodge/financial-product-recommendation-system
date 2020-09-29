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

app = Flask(__name__)
CORS(app)

model = load_model(Config.getNNModel(), compile=False)

# read file
df = DataStore.getNNPostProcessingFileInput()


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

    batch_size = 256
    # 2.47 ms
    desired_user_index = df['user_index'].max() + 1 if df[df[ds.USER_COLUMN] == desired_user].empty else df[df[ds.USER_COLUMN] == desired_user]['user_index'].to_numpy()[
        0]
    desired_age_index = df['age_index'].max() + 1 if df[df[ds.AGE_COLUMN]
                                                        == desired_age].empty else df[df[ds.AGE_COLUMN]
                                                                                      == desired_age]['age_index'].to_numpy()[0]
    desired_gender_index = df['gender_index'].max() + 1 if df[df[ds.GENDER_COLUMN]
                                                              == desired_gender].empty else df[df[ds.GENDER_COLUMN]
                                                                                               == desired_gender]['gender_index'].to_numpy()[0]
    desired_marital_status_index = df['marital_status_index'].max() + 1 if df[df[ds.MARITAL_STATUS_COLUMN]
                                                                              == desired_marital].empty else df[df[ds.MARITAL_STATUS_COLUMN]
                                                                                                                == desired_marital]['marital_status_index'].to_numpy()[0]

    desired_education_index = df['edu_level_index'].max() + 1 if df[df[ds.EDUCATION_LEVEL_COLUMN]
                                                                    == desired_edu_level].empty else df[df[ds.EDUCATION_LEVEL_COLUMN]
                                                                                                        == desired_edu_level]['edu_level_index'].to_numpy()[0]
    # 17.2 ms
    # # if user already process the product remove it from the list
    # # get a view on users processed products
    products_currently_owned_by_user = df[df['user_index'] ==
                                          desired_user_index]['symbol_index'].to_numpy()

    # get the products in similar users' portfolios
    products = []
    # potential_products = df[df[ds.USER_INDEX_COLUMN] != desired_user_index]
    # for po in products_currently_owned_by_user:
    #     potential_products = potential_products[potential_products[ds.PRODUCT_INDEX_COLUMN] != po]
    potential_products = df
    products += list(potential_products['symbol_index'])

    products = np.unique(products)

    modified_products = []
    for p in products:
        modified_product_element = []
        product_df = df[df['symbol_index'] == p]

        modified_product_element.append(
            product_df['symbol_index'].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.PRICE_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.FIVE_YEAR_AVERAGE_DIVIDEND_YIELD].to_numpy()[0])
        modified_product_element.append(
            product_df['sector_index'].to_numpy()[0])
        modified_product_element.append(
            product_df['industry_index'].to_numpy()[0])
        modified_product_element.append(1)
        modified_product_element.append(1)
        modified_product_element.append(1)
        modified_product_element.append(1)
        modified_products.append(modified_product_element)

    users = []
    for index in modified_products:
        user_element = []
        user_element.append(desired_user_index)
        user_element.append(desired_age_index)
        user_element.append(desired_gender_index)
        user_element.append(desired_marital_status_index)
        user_element.append(desired_education_index)
        user_element.append(desired_num_of_child)
        user_element.append(desired_risk_level)
        user_element.append(1)
        user_element.append(1)
        users.append(user_element)

    users = np.array(users)
    items = np.array(modified_products)

    # 22.1 ms
    print('\nRanking most likely products using the NeuMF model...')

    # and predict products for my user
    if len(products) > 0:
        results = model.predict(
            [users, items], batch_size=batch_size, verbose=0)
        # 207 ms
        results = results.tolist()

        results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
            ds.PROBABILITY_COLUMN, ds.SYMBOL_COLUMN])

        # loop through and get the probability (of being in the portfolio according to my model), the product name
        for i, prob in enumerate(results):
            results_df.loc[i] = [100 * prob[0], df[df['symbol_index'] == items[i][0]].iloc[0]
                                 [ds.SYMBOL_COLUMN]]
        results_df = results_df.sort_values(
            by=[ds.PROBABILITY_COLUMN], ascending=False)
    else:
        results_df = pd.DataFrame(np.nan, index=range(0), columns=[
            ds.PROBABILITY_COLUMN, ds.SYMBOL_COLUMN])

    return results_df.to_json(orient="records")
    # 209 ms


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

#     batch_size = 256

#     # 1.82 ms

#     desired_product_index = df[ds.PRODUCT_INDEX_COLUMN].max() + 1 if df[df[ds.PRODUCT_COLUMN] == desired_product].empty else df[df[ds.PRODUCT_COLUMN] == desired_product][ds.PRODUCT_INDEX_COLUMN].to_numpy()[
#         0]

#     desired_asset_class_index = df[ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].max() + 1 if df[df[ds.PRODUCT_ASSET_CLASS_COLUMN]
#                                                                                         == desired_asset_class].empty else df[df[ds.PRODUCT_ASSET_CLASS_COLUMN]
#                                                                                                                               == desired_asset_class][ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0]

#     users_currently_owned_by_product = df[df[ds.PRODUCT_INDEX_COLUMN] ==
#                                           desired_product_index][ds.USER_INDEX_COLUMN].to_numpy()

#     users_df = df[~df[ds.USER_INDEX_COLUMN].isin(
#         users_currently_owned_by_product)]

#     modified_users = users_df[[ds.USER_INDEX_COLUMN, ds.AGE_INDEX_COLUMN, ds.GENDER_INDEX_COLUMN,
#                                ds.MARITAL_STATUS_INDEX_COLUMN, ds.HAVE_CHILD_INDEX_COLUMN, ds.EDUCATION_INDEX_COLUMN]]

#     products = []
#     for index in range(len(modified_users)):
#         product_element = []
#         product_element.append(desired_product_index)
#         product_element.append(desired_3year_return)
#         product_element.append(desired_standard_deviation)
#         product_element.append(desired_dividend)
#         product_element.append(desired_asset_class_index)
#         product_element.append(1)
#         products.append(product_element)

#     users = np.array(modified_users)
#     items = np.array(products)

#     # 12.4 ms
#     print('\nRanking most likely products using the NeuMF model...')

#     # and predict products for my user
#     if len(users) > 0:
#         results = model.predict(
#             [users, items], batch_size=batch_size, verbose=0)
#         results = results.tolist()

#         results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
#             ds.PROBABILITY_COLUMN, ds.USER_COLUMN])

#         # loop through and get the probability (of being in the portfolio according to my model), the product name
#         for i, prob in enumerate(results):
#             results_df.loc[i] = [100 * prob[0], df[df[ds.USER_INDEX_COLUMN] == users[i][0]].iloc[0]
#                                  [ds.USER_COLUMN]]
#             results_df = results_df.sort_values(
#                 by=[ds.PROBABILITY_COLUMN], ascending=False)
#     else:
#         results_df = pd.DataFrame(np.nan, index=range(0), columns=[
#             ds.PROBABILITY_COLUMN, ds.USER_COLUMN])

#     return results_df.to_json(orient="records")

if __name__ == "__main__":
    app.run()
