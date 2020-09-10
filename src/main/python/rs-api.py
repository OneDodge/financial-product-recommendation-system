import tensorflow as tf
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

# custom class
from config import Config

app = Flask(__name__)


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

model = load_model(Config.getNNModel())

# read file
df = pd.read_csv(Config.getNNFileOutput(),
                 sep=",",
                 names=HEADERS,
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
                     PRODUCT_ASSET_CLASS_COLUMN: 'str',
                     AGE_CATEGORY_COLUMN: 'str',
                     USER_INDEX_COLUMN: np.int32,
                     AGE_INDEX_COLUMN: np.int32,
                     GENDER_INDEX_COLUMN: np.int32,
                     EDUCATION_INDEX_COLUMN: np.int32,
                     HAVE_CHILD_INDEX_COLUMN: np.int32,
                     MARITAL_STATUS_INDEX_COLUMN: np.int32,
                     PRODUCT_INDEX_COLUMN: np.int32,
                     PRODUCT_ASSET_CLASS_INDEX_COLUMN: np.int32
})


@app.route("/recommendation/product")
def productRecommendationApi():
    content = request.json

    desired_user = content["user"]
    desired_age = content["age"]
    desired_gender = content["gender"]
    desired_marital_status = content["maritalStatus"]
    desired_have_child = content["haveChild"]
    desired_education = content["education"]

    batch_size = 256

    desired_user_index = df[USER_INDEX_COLUMN].max() + 1 if df[df[USER_COLUMN] == desired_user].empty else df[df[USER_COLUMN] == desired_user][USER_INDEX_COLUMN].to_numpy()[
        0]
    desired_age_index = df[AGE_INDEX_COLUMN].max() + 1 if df[df[AGE_COLUMN]
                                                             == desired_age].empty else df[df[AGE_COLUMN]
                                                                                           == desired_age][AGE_INDEX_COLUMN].to_numpy()[0]
    desired_gender_index = df[GENDER_INDEX_COLUMN].max() + 1 if df[df[GENDER_COLUMN]
                                                                   == desired_gender].empty else df[df[GENDER_COLUMN]
                                                                                                    == desired_gender][GENDER_INDEX_COLUMN].to_numpy()[0]
    desired_marital_status_index = df[MARITAL_STATUS_INDEX_COLUMN].max() + 1 if df[df[MARITAL_STATUS_COLUMN]
                                                                                   == desired_marital_status].empty else df[df[MARITAL_STATUS_COLUMN]
                                                                                                                            == desired_marital_status][MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0]
    desired_have_child_index = df[HAVE_CHILD_INDEX_COLUMN].max() + 1 if df[df[HAVE_CHILD_COLUMN]
                                                                           == desired_have_child].empty else df[df[HAVE_CHILD_COLUMN]
                                                                                                                == desired_have_child][HAVE_CHILD_INDEX_COLUMN].to_numpy()[0]
    desired_education_index = df[EDUCATION_INDEX_COLUMN].max() + 1 if df[df[EDUCATION_COLUMN]
                                                                         == desired_education].empty else df[df[EDUCATION_COLUMN]
                                                                                                             == desired_education][EDUCATION_INDEX_COLUMN].to_numpy()[0]

    # # if user already process the product remove it from the list
    # # get a view on users processed products
    products_currently_owned_by_user = df[df[USER_INDEX_COLUMN] ==
                                          desired_user_index].product_index.to_numpy()

    # get the products in similar users' portfolios
    products = []
    # potential_products = df[df[USER_INDEX_COLUMN] != desired_user_index]
    # for po in products_currently_owned_by_user:
    #     potential_products = potential_products[potential_products[PRODUCT_INDEX_COLUMN] != po]
    potential_products = df
    products += list(potential_products[PRODUCT_INDEX_COLUMN])

    products = np.unique(products)

    modified_products = []
    for p in products:
        modified_product_element = []
        product_df = df[df[PRODUCT_INDEX_COLUMN] == p]

        modified_product_element.append(
            product_df[PRODUCT_INDEX_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0])
        modified_product_element.append(1)

        modified_products.append(modified_product_element)

    users = []
    for index in modified_products:
        user_element = []
        user_element.append(desired_user_index)
        user_element.append(desired_age_index)
        user_element.append(desired_gender_index)
        user_element.append(desired_marital_status_index)
        user_element.append(desired_have_child_index)
        user_element.append(desired_education_index)
        users.append(user_element)

    users = np.array(users)
    items = np.array(modified_products)

    print('\nRanking most likely products using the NeuMF model...')

    # and predict products for my user
    if len(products) > 0:
        results = model.predict(
            [users, items], batch_size=batch_size, verbose=0)
        results = results.tolist()

        results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
            PROBABILITY_COLUMN, PRODUCT_COLUMN])

        # loop through and get the probability (of being in the portfolio according to my model), the product name
        for i, prob in enumerate(results):
            results_df.loc[i] = [100 * prob[0], df[df[PRODUCT_INDEX_COLUMN] == items[i][0]].iloc[0]
                                 [PRODUCT_COLUMN]]
        results_df = results_df.sort_values(
            by=[PROBABILITY_COLUMN], ascending=False)
    else:
        results_df = pd.DataFrame(np.nan, index=range(0), columns=[
            PROBABILITY_COLUMN, PRODUCT_COLUMN])

    return results_df.to_json(orient="records")


@app.route("/recommendation/user")
def userRecommendationApi():
    content = request.json

    desired_product = content["product_name"]
    desired_3year_return = float(content["3year_return"])
    desired_standard_deviation = float(content["standard_deviation"])
    desired_dividend = float(content["dividend"])
    desired_asset_class = content["asset_class"]

    batch_size = 256

    desired_product_index = df[PRODUCT_INDEX_COLUMN].max() + 1 if df[df[PRODUCT_COLUMN] == desired_product].empty else df[df[PRODUCT_COLUMN] == desired_product][PRODUCT_INDEX_COLUMN].to_numpy()[
        0]

    desired_asset_class_index = df[PRODUCT_ASSET_CLASS_INDEX_COLUMN].max() + 1 if df[df[PRODUCT_ASSET_CLASS_COLUMN]
                                                                                     == desired_asset_class].empty else df[df[PRODUCT_ASSET_CLASS_COLUMN]
                                                                                                                           == desired_asset_class][PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0]
    # # if user already process the product remove it from the list
    # # get a view on users processed products
    users_currently_owned_by_product = df[df[PRODUCT_INDEX_COLUMN] ==
                                          desired_product_index][USER_INDEX_COLUMN].to_numpy()

    # # get the products in similar users' portfolios
    users = []
    potential_users = df[df[PRODUCT_INDEX_COLUMN] != desired_product_index]
    for po in users_currently_owned_by_product:
        potential_users = potential_users[potential_users[USER_INDEX_COLUMN] != po]
    # potential_users = df
    users += list(potential_users[USER_INDEX_COLUMN])

    users = np.unique(users)

    # print(len(users))
    users_df = df[df[USER_INDEX_COLUMN].isin(users)]
    modified_users = users_df[[USER_INDEX_COLUMN, AGE_INDEX_COLUMN, GENDER_INDEX_COLUMN,
                               MARITAL_STATUS_INDEX_COLUMN, HAVE_CHILD_INDEX_COLUMN, EDUCATION_INDEX_COLUMN]]
    # print(len(modified_users))
    # for u in users:
    #     modified_user_element = []
    #     user_df = df[df[USER_INDEX_COLUMN] == u]

    #     modified_user_element.append(
    #         user_df[USER_INDEX_COLUMN].to_numpy()[0])
    #     modified_user_element.append(
    #         user_df[AGE_INDEX_COLUMN].to_numpy()[0])
    #     modified_user_element.append(
    #         user_df[GENDER_INDEX_COLUMN].to_numpy()[0])
    #     modified_user_element.append(
    #         user_df[MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0])
    #     modified_user_element.append(
    #         user_df[HAVE_CHILD_INDEX_COLUMN].to_numpy()[0])
    #     modified_user_element.append(
    #         user_df[EDUCATION_INDEX_COLUMN].to_numpy()[0])
    #     modified_users.append(modified_user_element)

    products = []
    for index in range(len(modified_users)):
        product_element = []
        product_element.append(desired_product_index)
        product_element.append(desired_3year_return)
        product_element.append(desired_standard_deviation)
        product_element.append(desired_dividend)
        product_element.append(desired_asset_class_index)
        product_element.append(1)
        products.append(product_element)

    users = np.array(modified_users)
    items = np.array(products)

    print('\nRanking most likely products using the NeuMF model...')

    # and predict products for my user
    if len(users) > 0:
        results = model.predict(
            [users, items], batch_size=batch_size, verbose=0)
        results = results.tolist()

        results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
            PROBABILITY_COLUMN, USER_COLUMN])

        # loop through and get the probability (of being in the portfolio according to my model), the product name
        for i, prob in enumerate(results):
            results_df.loc[i] = [100 * prob[0], df[df[USER_INDEX_COLUMN] == users[i][0]].iloc[0]
                                 [USER_COLUMN]]
            results_df = results_df.sort_values(
                by=[PROBABILITY_COLUMN], ascending=False)
    else:
        results_df = pd.DataFrame(np.nan, index=range(0), columns=[
            PROBABILITY_COLUMN, USER_COLUMN])

    return results_df.to_json(orient="records")


@app.route("/recommendation/data")
def getData():
    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
