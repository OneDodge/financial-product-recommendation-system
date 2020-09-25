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

    desired_user = content["user"]
    desired_age = int(content["age"])
    desired_gender = content["gender"]
    desired_marital_status = content["maritalStatus"]
    desired_have_child = content["haveChild"]
    desired_education = content["education"]

    new_table = []
    result_table = []
    for product_name in df[ds.PRODUCT_COLUMN].unique():
        product_df = df[df[ds.PRODUCT_COLUMN] == product_name].iloc[[0]]

        new_row = []
        new_row.append(desired_age)
        new_row.append(desired_gender)
        new_row.append(desired_marital_status)
        new_row.append(desired_have_child)
        new_row.append(desired_education)
        new_row.append(product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[
            0])
        new_row.append(product_df[ds.PRODUCT_ASSET_CLASS_COLUMN].to_numpy()[
            0])

        new_table.append(new_row)
        result_item = []
        result_item.append(product_df[ds.PRODUCT_COLUMN].to_numpy()[
            0])
        result_table.append(result_item)

    new_table_df = pd.DataFrame(data=np.array(new_table))
    new_table_df.columns = [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.HAVE_CHILD_COLUMN, ds.EDUCATION_COLUMN,
                            ds.PRODUCT_3_YR_RETURN_COLUMN, ds.PRODUCT_STD_DEV_COLUMN, ds.PRODUCT_DEVIDEND_COLUMN, ds.PRODUCT_ASSET_CLASS_COLUMN]

    new_table_df[ds.AGE_COLUMN] = new_table_df[ds.AGE_COLUMN].astype(
        str).astype(int)
    new_table_df[ds.PRODUCT_3_YR_RETURN_COLUMN] = new_table_df[ds.PRODUCT_3_YR_RETURN_COLUMN].astype(
        str).astype(float)
    new_table_df[ds.PRODUCT_STD_DEV_COLUMN] = new_table_df[ds.PRODUCT_STD_DEV_COLUMN].astype(
        str).astype(float)
    new_table_df[ds.PRODUCT_DEVIDEND_COLUMN] = new_table_df[ds.PRODUCT_DEVIDEND_COLUMN].astype(
        str).astype(float)

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
    result_df.columns = [ds.PRODUCT_COLUMN, ds.PROBABILITY_COLUMN]
    result_df = result_df.sort_values(
        by=[ds.PROBABILITY_COLUMN], ascending=False)
    return result_df.to_json(orient="records")


@app.route("/recommendation/user", methods=['GET', 'POST'])
def userRecommendationApi():
    content = request.json

    desired_product = content["product_name"]
    desired_3year_return = float(content["3year_return"])
    desired_standard_deviation = float(content["standard_deviation"])
    desired_dividend = float(content["dividend"])
    desired_asset_class = content["asset_class"]

    new_table = []
    result_table = []
    for user_name in df[ds.USER_COLUMN].unique():
        user_df = df[df[ds.USER_COLUMN] == user_name].iloc[[0]]

        new_row = []
        new_row.append(user_df[ds.AGE_COLUMN].to_numpy()[
            0])
        new_row.append(user_df[ds.GENDER_COLUMN].to_numpy()[
            0])
        new_row.append(user_df[ds.MARITAL_STATUS_COLUMN].to_numpy()[
            0])
        new_row.append(user_df[ds.HAVE_CHILD_COLUMN].to_numpy()[
            0])
        new_row.append(user_df[ds.EDUCATION_COLUMN].to_numpy()[
            0])
        new_row.append(desired_3year_return)
        new_row.append(desired_standard_deviation)
        new_row.append(desired_dividend)
        new_row.append(desired_asset_class)

        new_table.append(new_row)
        result_item = []
        result_item.append(user_df[ds.USER_COLUMN].to_numpy()[
            0])
        result_table.append(result_item)

    new_table_df = pd.DataFrame(data=np.array(new_table))
    new_table_df.columns = [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.HAVE_CHILD_COLUMN, ds.EDUCATION_COLUMN,
                            ds.PRODUCT_3_YR_RETURN_COLUMN, ds.PRODUCT_STD_DEV_COLUMN, ds.PRODUCT_DEVIDEND_COLUMN, ds.PRODUCT_ASSET_CLASS_COLUMN]

    new_table_df[ds.AGE_COLUMN] = new_table_df[ds.AGE_COLUMN].astype(
        str).astype(int)
    new_table_df[ds.PRODUCT_3_YR_RETURN_COLUMN] = new_table_df[ds.PRODUCT_3_YR_RETURN_COLUMN].astype(
        str).astype(float)
    new_table_df[ds.PRODUCT_STD_DEV_COLUMN] = new_table_df[ds.PRODUCT_STD_DEV_COLUMN].astype(
        str).astype(float)
    new_table_df[ds.PRODUCT_DEVIDEND_COLUMN] = new_table_df[ds.PRODUCT_DEVIDEND_COLUMN].astype(
        str).astype(float)

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
    result_df.columns = [ds.USER_COLUMN, ds.PROBABILITY_COLUMN]
    result_df = result_df.sort_values(
        by=[ds.PROBABILITY_COLUMN], ascending=False)
    return result_df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
