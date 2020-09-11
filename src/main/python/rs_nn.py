
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
import rs_ds as ds
from rs_ds import DataStore

FLAGS = flags.FLAGS

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


def main():
    # read file
    df = df = DataStore.getNNFileInput()

    age_bins = [0, 18, 38, 58, 78, 98, np.inf]

    age_names = ['<18', '18-38', '38-58', '58-78', '78-98', '98+']

    df[ds.AGE_CATEGORY_COLUMN] = pd.cut(
        df[ds.AGE_COLUMN], age_bins, labels=age_names)

    # turn users and products into their unique cat codes so have a 0-N index for users and products
    df[ds.USER_INDEX_COLUMN] = df[ds.USER_COLUMN].astype('category').cat.codes
    df[ds.AGE_INDEX_COLUMN] = df[ds.AGE_CATEGORY_COLUMN].astype(
        'category').cat.codes
    df[ds.GENDER_INDEX_COLUMN] = df[ds.GENDER_COLUMN].astype(
        'category').cat.codes
    df[ds.EDUCATION_INDEX_COLUMN] = df[ds.EDUCATION_COLUMN].astype(
        'category').cat.codes
    df[ds.HAVE_CHILD_INDEX_COLUMN] = df[ds.HAVE_CHILD_COLUMN].astype(
        'category').cat.codes
    df[ds.MARITAL_STATUS_INDEX_COLUMN] = df[ds.MARITAL_STATUS_COLUMN].astype(
        'category').cat.codes
    df[ds.PRODUCT_INDEX_COLUMN] = df[ds.PRODUCT_COLUMN].astype(
        'category').cat.codes
    df[ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN] = df[ds.PRODUCT_ASSET_CLASS_COLUMN].astype(
        'category').cat.codes

    x_train = df[ds.USER_INDEX_COLUMN]
    y_train = df[ds.PRODUCT_INDEX_COLUMN]

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
        user_df = df[df[ds.USER_INDEX_COLUMN] == ui]
        modified_x_train_element = []
        modified_x_train_element.append(ui)
        modified_x_train_element.append(
            user_df[ds.AGE_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[ds.GENDER_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[ds.MARITAL_STATUS_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[ds.HAVE_CHILD_INDEX_COLUMN].to_numpy()[0])
        modified_x_train_element.append(
            user_df[ds.EDUCATION_INDEX_COLUMN].to_numpy()[0])
        modified_x_train.append(modified_x_train_element)

    modified_y_train = []
    for ii in item_input_train:
        product_df = df[df[ds.PRODUCT_INDEX_COLUMN] == ii]
        modified_y_train_element = []
        modified_y_train_element.append(ii)
        modified_y_train_element.append(
            product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
        modified_y_train_element.append(
            product_df[ds.PRODUCT_ASSET_CLASS_INDEX_COLUMN].to_numpy()[0])
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
