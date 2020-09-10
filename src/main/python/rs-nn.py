
import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import HDFStore

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Embedding, Concatenate, Multiply, Lambda
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import scipy.sparse as sp
import argparse
from absl import flags
from sklearn.model_selection import train_test_split
import random
import os
# import ssl

# custom class
from config import Config


FLAGS = flags.FLAGS

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
           PRODUCT_COLUMN, PRODUCT_3_YR_RETURN_COLUMN, PRODUCT_STD_DEV_COLUMN, PRODUCT_DEVIDEND_COLUMN, PRODUCT_ASSET_CLASS_COLUMN]

PROBABILITY_COLUMN = 'probability (%)'

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)


def get_model(num_users, num_items, latent_dim=8, dense_layers=[64, 32, 16, 8],
              reg_layers=[0, 0, 0, 0], reg_mf=0):

    # input layer
    input_user = Input(shape=(6,), dtype='float64', name='user_input')
    input_item = Input(shape=(6,), dtype='float64', name='item_input')

    # embedding layer
    mf_user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim,
                                  name='mf_user_embedding',
                                  embeddings_initializer='glorot_uniform',
                                  embeddings_regularizer=l2(reg_mf),
                                  input_length=1)
    mf_item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim,
                                  name='mf_item_embedding',
                                  embeddings_initializer='glorot_uniform',
                                  embeddings_regularizer=l2(reg_mf),
                                  input_length=1)
    mlp_user_embedding = Embedding(input_dim=num_users, output_dim=int(dense_layers[0]/2),
                                   name='mlp_user_embedding',
                                   embeddings_initializer='glorot_uniform',
                                   embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    mlp_item_embedding = Embedding(input_dim=num_items, output_dim=int(dense_layers[0]/2),
                                   name='mlp_item_embedding',
                                   embeddings_initializer='glorot_uniform',
                                   embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # MF latent vector
    mf_user_latent = Flatten()(mf_user_embedding(input_user))
    mf_item_latent = Flatten()(mf_item_embedding(input_item))
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # MLP latent vector
    mlp_user_latent = Flatten()(mlp_user_embedding(input_user))
    mlp_item_latent = Flatten()(mlp_item_embedding(input_item))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layer for model
    for i in range(1, len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    # result = Dense(1, activation='softmax',
    #                kernel_initializer='lecun_uniform', name='result')
    result = Dense(1, activation='sigmoid',
                   kernel_initializer='lecun_uniform', name='result')

    model = Model(inputs=[input_user, input_item],
                  outputs=result(predict_layer))
    return model

# get the training samples


def get_train_samples(train_mat, num_negatives):
    user_input, item_input, labels = [], [], []
    num_user, num_item = train_mat.shape
    for (u, i) in train_mat.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        negative_sampling_list = random.sample(
            range(num_item), min(num_negatives, num_item))
        for j in negative_sampling_list:
            if (u, j) not in train_mat.keys():
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
    return user_input, item_input, labels

# An example of the recommendations for financial products


def printDataFrameDistribution(df, product_name):
    desired_product_name = product_name
    desired_product_df = df[df[PRODUCT_COLUMN] == desired_product_name]
    # Print each columns distribution
    print("-----------General(%s)-------------------------------------" %
          (product_name))
    print(desired_product_df)
    print("-----------Age Distribution(%s)----------------------------" %
          (product_name))
    print(desired_product_df.groupby(
        [AGE_CATEGORY_COLUMN]).size())
    print(desired_product_df.groupby(
        [AGE_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Gender Distribution(%s)-------------------------" %
          (product_name))
    print(desired_product_df.groupby(
        [GENDER_COLUMN, GENDER_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Education Distribution(%s)----------------------" %
          (product_name))
    print(desired_product_df.groupby(
        [EDUCATION_COLUMN, EDUCATION_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Have Child Distribution(%s)---------------------" %
          (product_name))
    print(desired_product_df.groupby(
        [HAVE_CHILD_COLUMN, HAVE_CHILD_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Marital Status Distribution(%s)-----------------" %
          (product_name))
    print(desired_product_df.groupby(
        [MARITAL_STATUS_COLUMN, MARITAL_STATUS_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Product Distribution(%s)------------------------" %
          (product_name))
    print(desired_product_df.groupby(
        [PRODUCT_COLUMN, PRODUCT_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")
    print("-----------Product Asset Class Distribution(%s)------------" %
          (product_name))
    print(desired_product_df.groupby(
        [PRODUCT_ASSET_CLASS_COLUMN, PRODUCT_ASSET_CLASS_INDEX_COLUMN]).size())
    print("-----------------------------------------------------------")


def main():
    # read file
    df = pd.read_csv(Config.getNNFileInput(),
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
                         PRODUCT_ASSET_CLASS_COLUMN: 'str'
    })

    age_bins = [0, 18, 38, 58, 78, 98, np.inf]

    age_names = ['<18', '18-38', '38-58', '58-78', '78-98', '98+']

    df[AGE_CATEGORY_COLUMN] = pd.cut(
        df[AGE_COLUMN], age_bins, labels=age_names)

    # turn users and products into their unique cat codes so have a 0-N index for users and products
    df[USER_INDEX_COLUMN] = df[USER_COLUMN].astype('category').cat.codes
    df[AGE_INDEX_COLUMN] = df[AGE_CATEGORY_COLUMN].astype(
        'category').cat.codes
    df[GENDER_INDEX_COLUMN] = df[GENDER_COLUMN].astype(
        'category').cat.codes
    df[EDUCATION_INDEX_COLUMN] = df[EDUCATION_COLUMN].astype(
        'category').cat.codes
    df[HAVE_CHILD_INDEX_COLUMN] = df[HAVE_CHILD_COLUMN].astype(
        'category').cat.codes
    df[MARITAL_STATUS_INDEX_COLUMN] = df[MARITAL_STATUS_COLUMN].astype(
        'category').cat.codes
    df[PRODUCT_INDEX_COLUMN] = df[PRODUCT_COLUMN].astype(
        'category').cat.codes
    df[PRODUCT_ASSET_CLASS_INDEX_COLUMN] = df[PRODUCT_ASSET_CLASS_COLUMN].astype(
        'category').cat.codes

    # Print each columns distribution
    print("-----------General------------")
    print(df)
    print("-----------Age Distribution------------")
    print(df.groupby(
        [AGE_CATEGORY_COLUMN]).size())
    print(df.groupby(
        [AGE_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Gender Distribution------------")
    print(df.groupby(
        [GENDER_COLUMN, GENDER_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Education Distribution----------")
    print(df.groupby(
        [EDUCATION_COLUMN, EDUCATION_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Have Child Distribution---------")
    print(df.groupby(
        [HAVE_CHILD_COLUMN, HAVE_CHILD_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Marital Status Distribution-----")
    print(df.groupby(
        [MARITAL_STATUS_COLUMN, MARITAL_STATUS_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Product Distribution------------")
    print(df.groupby(
        [PRODUCT_COLUMN, PRODUCT_INDEX_COLUMN]).size())
    print("-------------------------------------------")
    print("-----------Product Asset Class Distribution------------")
    print(df.groupby(
        [PRODUCT_ASSET_CLASS_COLUMN, PRODUCT_ASSET_CLASS_INDEX_COLUMN]).size())
    print("-------------------------------------------")

    x_train = df[USER_INDEX_COLUMN]
    y_train = df[PRODUCT_INDEX_COLUMN]

    # save data in dok matrix(optimized sparse matrix object)
    # create a sparse portfolio x products matrix
    # if a user i has product j, mat[i, j] = 1
    mat_train = sp.dok_matrix((x_train.shape[0], len(
        y_train.unique())), dtype=np.float32)
    for userIndex, productIndex in zip(x_train, y_train):
        mat_train[userIndex, productIndex] = 1

    # hyperparameters
    # loaded = True
    verbose = 1
    epochs = 40
    batch_size = 256
    latent_dim = 64
    # dense_layers = [64, 32, 16, 8]
    # reg_layers = [0, 0, 0, 0]
    # dense_layers = [64, 32, 16, 8]
    # reg_layers = [0.000001, 0.000001, 0.000001, 0.000001]
    dense_layers = [512, 256, 128, 64, 32, 16, 8]
    reg_layers = [0.000001, 0.000001, 0.000001,
                  0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
    reg_mf = 0.000001
    # reg_mf = 0
    num_negatives = 2
    learning_rate = 0.001
    # learner = 'adam'
    # dataset = 'portfolio'

    num_users, num_items = mat_train.shape
    # # # print('Done loading data!')

    # # contstruct the NeuMF Neural Network
    model = get_model(num_users+1, (num_items*3)+1, latent_dim,
                      dense_layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    print(model.summary())

    user_input_train, item_input_train, labels_train = get_train_samples(
        mat_train, num_negatives)

    # organise training input<User,Product><Label>
    modified_x_train = []
    for ui in user_input_train:
        user_df = df[df[USER_INDEX_COLUMN] == ui]
        modified_x_train_element = []
        modified_x_train_element.append(ui)
        modified_x_train_element.append(
            user_df[AGE_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[GENDER_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[HAVE_CHILD_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[EDUCATION_INDEX_COLUMN].to_numpy()[0])
        modified_x_train.append(modified_x_train_element)

    modified_y_train = []
    for ii in item_input_train:
        product_df = df[df[PRODUCT_INDEX_COLUMN] == ii]
        modified_y_train_element = []
        modified_y_train_element.append(ii)
        modified_y_train_element.append(
            product_df[PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0])
        modified_y_train_element.append(1)
        modified_y_train.append(modified_y_train_element)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Config.getNNCheckpoint(),
                                                     save_weights_only=True,
                                                     verbose=0)

    hist = model.fit([np.array(modified_x_train), np.array(modified_y_train)], np.array(labels_train),
                     batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True, callbacks=[cp_callback])

    model.save(Config.getNNModel())

    df.to_csv(Config.getNNFileOutput(), index=False)


if __name__ == '__main__':
    main()
