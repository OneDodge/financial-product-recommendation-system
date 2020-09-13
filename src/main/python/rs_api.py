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
df = DataStore.getNNFileOutput()


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
    # 2.47 ms
    desired_user_index = df[ds.USER_INDEX_COLUMN].max() + 1 if df[df[ds.USER_COLUMN] == desired_user].empty else df[df[ds.USER_COLUMN] == desired_user][ds.USER_INDEX_COLUMN].to_numpy()[
        0]
    desired_age_index = df[ds.AGE_INDEX_COLUMN].max() + 1 if df[df[ds.AGE_COLUMN]
                                                                == desired_age].empty else df[df[ds.AGE_COLUMN]
                                                                                              == desired_age][ds.AGE_INDEX_COLUMN].to_numpy()[0]
    desired_gender_index = df[ds.GENDER_INDEX_COLUMN].max() + 1 if df[df[ds.GENDER_COLUMN]
                                                                      == desired_gender].empty else df[df[ds.GENDER_COLUMN]
                                                                                                       == desired_gender][ds.GENDER_INDEX_COLUMN].to_numpy()[0]
    desired_marital_status_index = df[ds.MARITAL_STATUS_INDEX_COLUMN].max() + 1 if df[df[ds.MARITAL_STATUS_COLUMN]
                                                                                      == desired_marital_status].empty else df[df[ds.MARITAL_STATUS_COLUMN]
                                                                                                                               == desired_marital_status][ds.MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0]
    desired_have_child_index = df[ds.HAVE_CHILD_INDEX_COLUMN].max() + 1 if df[df[ds.HAVE_CHILD_COLUMN]
                                                                              == desired_have_child].empty else df[df[ds.HAVE_CHILD_COLUMN]
                                                                                                                   == desired_have_child][ds.HAVE_CHILD_INDEX_COLUMN].to_numpy()[0]
    desired_education_index = df[ds.EDUCATION_INDEX_COLUMN].max() + 1 if df[df[ds.EDUCATION_COLUMN]
                                                                            == desired_education].empty else df[df[ds.EDUCATION_COLUMN]
                                                                                                                == desired_education][ds.EDUCATION_INDEX_COLUMN].to_numpy()[0]
    # 17.2 ms
    # # if user already process the product remove it from the list
    # # get a view on users processed products
    products_currently_owned_by_user = df[df[ds.USER_INDEX_COLUMN] ==
                                          desired_user_index][ds.PRODUCT_INDEX_COLUMN].to_numpy()

    # get the products in similar users' portfolios
    products = []
    # potential_products = df[df[ds.USER_INDEX_COLUMN] != desired_user_index]
    # for po in products_currently_owned_by_user:
    #     potential_products = potential_products[potential_products[ds.PRODUCT_INDEX_COLUMN] != po]
    potential_products = df
    products += list(potential_products[ds.PRODUCT_INDEX_COLUMN])

    products = np.unique(products)

    modified_products = []
    for p in products:
        modified_product_element = []
        product_df = df[df[ds.PRODUCT_INDEX_COLUMN] == p]

        modified_product_element.append(
            product_df[ds.PRODUCT_INDEX_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
        modified_product_element.append(
            product_df[ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0])
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

    # 22.1 ms
    print('\nRanking most likely products using the NeuMF model...')

    # and predict products for my user
    if len(products) > 0:
        results = model.predict(
            [users, items], batch_size=batch_size, verbose=0)
        # 207 ms
        results = results.tolist()

        results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
            ds.PROBABILITY_COLUMN, ds.PRODUCT_COLUMN])

        # loop through and get the probability (of being in the portfolio according to my model), the product name
        for i, prob in enumerate(results):
            results_df.loc[i] = [100 * prob[0], df[df[ds.PRODUCT_INDEX_COLUMN] == items[i][0]].iloc[0]
                                 [ds.PRODUCT_COLUMN]]
        results_df = results_df.sort_values(
            by=[ds.PROBABILITY_COLUMN], ascending=False)
    else:
        results_df = pd.DataFrame(np.nan, index=range(0), columns=[
            ds.PROBABILITY_COLUMN, ds.PRODUCT_COLUMN])

    return results_df.to_json(orient="records")
    # 209 ms


@app.route("/recommendation/user")
def userRecommendationApi():
    content = request.json

    desired_product = content["product_name"]
    desired_3year_return = float(content["3year_return"])
    desired_standard_deviation = float(content["standard_deviation"])
    desired_dividend = float(content["dividend"])
    desired_asset_class = content["asset_class"]

    batch_size = 256

    # 1.82 ms

    desired_product_index = df[ds.PRODUCT_INDEX_COLUMN].max() + 1 if df[df[ds.PRODUCT_COLUMN] == desired_product].empty else df[df[ds.PRODUCT_COLUMN] == desired_product][ds.PRODUCT_INDEX_COLUMN].to_numpy()[
        0]

    desired_asset_class_index = df[ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].max() + 1 if df[df[ds.PRODUCT_ASSET_CLASS_COLUMN]
                                                                                        == desired_asset_class].empty else df[df[ds.PRODUCT_ASSET_CLASS_COLUMN]
                                                                                                                              == desired_asset_class][ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0]

    users_currently_owned_by_product = df[df[ds.PRODUCT_INDEX_COLUMN] ==
                                          desired_product_index][ds.USER_INDEX_COLUMN].to_numpy()

    users_df = df[~df[ds.USER_INDEX_COLUMN].isin(
        users_currently_owned_by_product)]

    modified_users = users_df[[ds.USER_INDEX_COLUMN, ds.AGE_INDEX_COLUMN, ds.GENDER_INDEX_COLUMN,
                               ds.MARITAL_STATUS_INDEX_COLUMN, ds.HAVE_CHILD_INDEX_COLUMN, ds.EDUCATION_INDEX_COLUMN]]

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

    # 12.4 ms
    print('\nRanking most likely products using the NeuMF model...')

    # and predict products for my user
    if len(users) > 0:
        results = model.predict(
            [users, items], batch_size=batch_size, verbose=0)
        results = results.tolist()

        results_df = pd.DataFrame(np.nan, index=range(len(results)), columns=[
            ds.PROBABILITY_COLUMN, ds.USER_COLUMN])

        # loop through and get the probability (of being in the portfolio according to my model), the product name
        for i, prob in enumerate(results):
            results_df.loc[i] = [100 * prob[0], df[df[ds.USER_INDEX_COLUMN] == users[i][0]].iloc[0]
                                 [ds.USER_COLUMN]]
            results_df = results_df.sort_values(
                by=[ds.PROBABILITY_COLUMN], ascending=False)
    else:
        results_df = pd.DataFrame(np.nan, index=range(0), columns=[
            ds.PROBABILITY_COLUMN, ds.USER_COLUMN])

    return results_df.to_json(orient="records")


@app.route("/recommendation/data")
def getData():
    result_df = df
    for k in request.values:
        val = int(request.values.get(k)) if request.values.get(
            k).isnumeric() else request.values.get(k)
        result_df = result_df[result_df[k] == val]
    return result_df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
