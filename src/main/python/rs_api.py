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

        new_row = {}
        new_row[ds.AGE_COLUMN] = desired_age
        new_row[ds.GENDER_COLUMN] = desired_gender
        new_row[ds.MARITAL_STATUS_COLUMN] = desired_marital_status
        new_row[ds.HAVE_CHILD_COLUMN] = desired_have_child
        new_row[ds.EDUCATION_COLUMN] = desired_education
        new_row[ds.PRODUCT_3_YR_RETURN_COLUMN] = product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[
            0]
        new_row[ds.PRODUCT_STD_DEV_COLUMN] = product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[
            0]
        new_row[ds.PRODUCT_DEVIDEND_COLUMN] = product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[
            0]
        new_row[ds.PRODUCT_ASSET_CLASS_COLUMN] = product_df[ds.PRODUCT_ASSET_CLASS_COLUMN].to_numpy()[
            0]

        new_table.append(new_row)
        result_item = []
        result_item.append(product_df[ds.PRODUCT_COLUMN].to_numpy()[
            0])
        result_table.append(result_item)

    for i in range(len(new_table)):
        input_dict = {name: tf.convert_to_tensor(
            [value]) for name, value in new_table[i].items()}
        predictions = model.predict(input_dict)

        result_item = result_table[i]
        result_item.append(predictions[0][0] * 100)
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

        new_row = {}
        new_row[ds.AGE_COLUMN] = user_df[ds.AGE_COLUMN].to_numpy()[
            0]
        new_row[ds.GENDER_COLUMN] = user_df[ds.GENDER_COLUMN].to_numpy()[
            0]
        new_row[ds.MARITAL_STATUS_COLUMN] = user_df[ds.MARITAL_STATUS_COLUMN].to_numpy()[
            0]
        new_row[ds.HAVE_CHILD_COLUMN] = user_df[ds.HAVE_CHILD_COLUMN].to_numpy()[
            0]
        new_row[ds.EDUCATION_COLUMN] = user_df[ds.EDUCATION_COLUMN].to_numpy()[
            0]
        new_row[ds.PRODUCT_3_YR_RETURN_COLUMN] = desired_3year_return
        new_row[ds.PRODUCT_STD_DEV_COLUMN] = desired_standard_deviation
        new_row[ds.PRODUCT_DEVIDEND_COLUMN] = desired_dividend
        new_row[ds.PRODUCT_ASSET_CLASS_COLUMN] = desired_asset_class

        new_table.append(new_row)
        result_item = []
        result_item.append(user_df[ds.USER_COLUMN].to_numpy()[
            0])
        result_table.append(result_item)

    for i in range(len(new_table)):
        input_dict = {name: tf.convert_to_tensor(
            [value]) for name, value in new_table[i].items()}
        predictions = model.predict(input_dict)

        result_item = result_table[i]
        result_item.append(predictions[0][0] * 100)
        result_table[i] = result_item
    result_table = np.array(result_table)
    result_df = pd.DataFrame(data=result_table)
    result_df.columns = [ds.USER_COLUMN, ds.PROBABILITY_COLUMN]
    result_df = result_df.sort_values(
        by=[ds.PROBABILITY_COLUMN], ascending=False)
    return result_df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
