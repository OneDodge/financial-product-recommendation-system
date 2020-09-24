
import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import HDFStore

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Embedding, Concatenate, Multiply, Lambda
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, layers
from tensorflow import feature_column

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
tf.keras.backend.set_floatx('float64')

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def get_model(all_user_features, all_product_features, num_users, num_items, latent_dim=8, dense_layers=[64, 32, 16, 8],
              reg_layers=[0, 0, 0, 0], reg_mf=0):

    # input layer
    # input_user = Input(shape=(5,), dtype='float64', name='user_input')
    # input_item = Input(shape=(5,), dtype='float64', name='item_input')

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
    # mf_user_latent = Flatten()(mf_user_embedding(user_feature_layer_outputs))
    # mf_item_latent = Flatten()(mf_item_embedding(product_feature_layer_outputs))
    # mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])

    # MLP latent vector
    mlp_user_latent = Flatten()(mlp_user_embedding(all_user_features))
    mlp_item_latent = Flatten()(mlp_item_embedding(all_product_features))
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layer for model
    for i in range(1, len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    # predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    # result = Dense(1, activation='softmax',
    #                kernel_initializer='lecun_uniform', name='result')
    result = Dense(1, activation='sigmoid',
                   kernel_initializer='lecun_uniform', name='result')

    model = Model(inputs=[all_user_features, all_product_features],
                  outputs=result(mlp_vector))
    return model

# get the training samples


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices. The lambda function captures the
    # layer so we can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


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


TARGET = 'target'


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(TARGET)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def main():

    # hyperparameters
    verbose = 1
    epochs = 40
    batch_size = 256
    latent_dim = 64
    dense_layers = [512, 256, 128, 64, 32, 16, 8]
    reg_layers = [0.000001, 0.000001, 0.000001,
                  0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
    reg_mf = 0.000001
    num_negatives = 2
    learning_rate = 0.001

    # read file
    df = DataStore.getNNFileInput()

    df[ds.USER_INDEX_COLUMN] = df[ds.USER_COLUMN].astype('category').cat.codes
    df[ds.PRODUCT_INDEX_COLUMN] = df[ds.PRODUCT_COLUMN].astype(
        'category').cat.codes

    df[TARGET] = 1

    x_train = df[ds.USER_INDEX_COLUMN]
    y_train = df[ds.PRODUCT_INDEX_COLUMN]

    # save data in dok matrix(optimized sparse matrix object)
    # create a sparse portfolio x products matrix
    # if a user i has product j, mat[i, j] = 1
    mat_train = sp.dok_matrix((x_train.shape[0], len(
        y_train.unique())), dtype=np.float32)
    for userIndex, productIndex in zip(x_train, y_train):
        mat_train[userIndex, productIndex] = 1
    user_input_train, item_input_train, labels_train = get_train_samples(
        mat_train, num_negatives)

    # print(user_input_train)
    # print(item_input_train)
    # print(labels_train)

    new_table = []
    for i in range(len(user_input_train)):
        user_df = df[df[ds.USER_INDEX_COLUMN] == user_input_train[i]]
        product_df = df[df[ds.PRODUCT_INDEX_COLUMN]
                        == item_input_train[i]].iloc[[0]]
        label = labels_train[i]

        new_row = []
        new_row.append(user_df[ds.AGE_COLUMN].to_numpy()[0])
        new_row.append(user_df[ds.GENDER_COLUMN].to_numpy()[0])
        new_row.append(user_df[ds.MARITAL_STATUS_COLUMN].to_numpy()[0])
        new_row.append(user_df[ds.HAVE_CHILD_COLUMN].to_numpy()[0])
        new_row.append(user_df[ds.EDUCATION_COLUMN].to_numpy()[0])
        new_row.append(
            product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
        new_row.append(
            product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
        new_row.append(
            product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
        new_row.append(product_df[ds.PRODUCT_ASSET_CLASS_COLUMN].to_numpy()[0])
        new_row.append(label)

        new_table.append(new_row)

    new_table = np.array(new_table)
    df = pd.DataFrame(data=new_table)
    df.columns = [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.HAVE_CHILD_COLUMN, ds.EDUCATION_COLUMN,
                  ds.PRODUCT_3_YR_RETURN_COLUMN, ds.PRODUCT_STD_DEV_COLUMN, ds.PRODUCT_DEVIDEND_COLUMN, ds.PRODUCT_ASSET_CLASS_COLUMN, TARGET]

    df[ds.AGE_COLUMN] = df[ds.AGE_COLUMN].astype(str).astype(int)
    df[ds.PRODUCT_3_YR_RETURN_COLUMN] = df[ds.PRODUCT_3_YR_RETURN_COLUMN].astype(
        str).astype(float)
    df[ds.PRODUCT_STD_DEV_COLUMN] = df[ds.PRODUCT_STD_DEV_COLUMN].astype(
        str).astype(float)
    df[ds.PRODUCT_DEVIDEND_COLUMN] = df[ds.PRODUCT_DEVIDEND_COLUMN].astype(
        str).astype(float)
    df[TARGET] = df[TARGET].astype(str).astype(int)
    # print(new_table)
    # df = df.drop(
    #     columns=[ds.USER_COLUMN, ds.PRODUCT_COLUMN, ds.USER_INDEX_COLUMN, ds.PRODUCT_INDEX_COLUMN])

    print(df)
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    all_user_features_input = []
    user_encoded_features = []

    all_product_features_input = []
    product_encoded_features = []

    all_inputs = []
    encoded_features = []

    batch_size = 256
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    print("------------- train ds")
    print(train_ds)
    # Users
    for header in [ds.AGE_COLUMN]:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_user_features_input.append(numeric_col)
        user_encoded_features.append(encoded_numeric_col)

        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    for header in [ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.HAVE_CHILD_COLUMN, ds.EDUCATION_COLUMN]:
        categorical_col = tf.keras.Input(
            shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=None)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_user_features_input.append(categorical_col)
        user_encoded_features.append(encoded_categorical_col)

        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    # Products
    for header in [ds.PRODUCT_3_YR_RETURN_COLUMN, ds.PRODUCT_STD_DEV_COLUMN, ds.PRODUCT_DEVIDEND_COLUMN]:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_product_features_input.append(numeric_col)
        product_encoded_features.append(encoded_numeric_col)

        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    for header in [ds.PRODUCT_ASSET_CLASS_COLUMN]:
        categorical_col = tf.keras.Input(
            shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=None)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_product_features_input.append(categorical_col)
        product_encoded_features.append(encoded_categorical_col)

        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_user_features = tf.keras.layers.concatenate(user_encoded_features)
    all_product_features = tf.keras.layers.concatenate(
        product_encoded_features)
    all_features = tf.keras.layers.concatenate(encoded_features)

    print(all_user_features)
    print(all_product_features)
    print(all_features)

    num_users = 999999
    num_items = 999999

    # mlp_user_embedding = Embedding(input_dim=num_users, output_dim=int(dense_layers[0]/3),
    #                                name='mlp_user_embedding',
    #                                embeddings_initializer='glorot_uniform',
    #                                embeddings_regularizer=l2(reg_layers[0]),
    #                                input_length=1)
    # mlp_item_embedding = Embedding(input_dim=num_items, output_dim=int(dense_layers[0]/2),
    #                                name='mlp_item_embedding',
    #                                embeddings_initializer='glorot_uniform',
    #                                embeddings_regularizer=l2(reg_layers[0]),
    #                                input_length=1)

    mlp_user_latent = Flatten()(all_user_features)
    mlp_item_latent = Flatten()(all_product_features)
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    mlp_vector = mlp_cat_latent

    # build dense layer for model
    for i in range(0, len(dense_layers)):
        layer = Dense(dense_layers[i],
                      activity_regularizer=l2(reg_layers[i]),
                      activation='relu',
                      name='layer%d' % i)
        mlp_vector = layer(mlp_vector)

    # # predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    # # result = Dense(1, activation='softmax',
    # #                kernel_initializer='lecun_uniform', name='result')
    output = Dense(1, activation='sigmoid',
                   kernel_initializer='lecun_uniform', name='result')(mlp_vector)

    # model = Model(inputs=[all_user_features, all_product_features],
    #               outputs=result(mlp_vector))

    # x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # output = tf.keras.layers.Dense(1)(x)

    # model = tf.keras.Model(all_inputs, output)
    model = Model(inputs=all_inputs,
                  outputs=output)

    # loss, accuracy = model.evaluate(test_ds)

    # print("Accuracy", accuracy)

    # print("train_ds")
    # print(train_ds)

    # for feature_batch, label_batch in train_ds.take(1):
    #     print('Every feature:', list(feature_batch.keys()))
    #     print('A batch of ages:', feature_batch[ds.AGE_COLUMN])
    #     print('A batch of targets:', label_batch)

    # user_feature_columns = []
    # user_feature_layer_inputs = {}

    # product_feature_columns = []
    # product_feature_layer_inputs = {}

    # for user_header in [ds.AGE_COLUMN, ds.GENDER_COLUMN, ds.MARITAL_STATUS_COLUMN, ds.HAVE_CHILD_COLUMN, ds.EDUCATION_COLUMN]:
    #     user_feature_columns.append(feature_column.numeric_column(user_header))
    #     user_feature_layer_inputs[user_header] = tf.keras.Input(
    #         shape=(1,), name=user_header)

    # for product_header in [ds.PRODUCT_3_YR_RETURN_COLUMN, ds.PRODUCT_STD_DEV_COLUMN, ds.PRODUCT_DEVIDEND_COLUMN, ds.PRODUCT_ASSET_CLASS_COLUMN, 'default']:
    #     product_feature_columns.append(
    #         feature_column.numeric_column(product_header))
    #     product_feature_layer_inputs[product_header] = tf.keras.Input(
    #         shape=(1,), name=product_header)

    # print(user_feature_columns)
    # print(product_feature_columns)

    # example_batch = next(iter(train_ds))[0]

    # def demo(feature_column):
    #     feature_layer = layers.DenseFeatures(feature_column)
    #     print(feature_layer(example_batch).numpy())

    # age = feature_column.numeric_column(ds.AGE_COLUMN)
    # age_buckets = feature_column.bucketized_column(
    #     age, boundaries=[0, 18, 30, 44, 54, 90])
    # user_feature_columns.append(age_buckets)
    # user_feature_layer_inputs[ds.AGE_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.AGE_COLUMN, dtype=tf.int32)

    # education_type = feature_column.categorical_column_with_vocabulary_list(
    #     ds.EDUCATION_COLUMN, ['PRIMARY', 'SECONDARY', 'UNIVERSITY', 'POSTGRADUATE'])

    # education_type_one_hot = feature_column.indicator_column(education_type)
    # user_feature_columns.append(education_type_one_hot)
    # user_feature_layer_inputs[ds.EDUCATION_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.EDUCATION_COLUMN, dtype=tf.string)

    # gender_type = feature_column.categorical_column_with_vocabulary_list(
    #     ds.GENDER_COLUMN, ['M', 'F'])

    # gender_type_one_hot = feature_column.indicator_column(gender_type)
    # user_feature_columns.append(gender_type_one_hot)
    # user_feature_layer_inputs[ds.GENDER_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.GENDER_COLUMN, dtype=tf.string)

    # marital_status_type = feature_column.categorical_column_with_vocabulary_list(
    #     ds.MARITAL_STATUS_COLUMN, ['SINGLE', 'MARRIED', 'DIVORCED'])

    # marital_status_type_one_hot = feature_column.indicator_column(
    #     marital_status_type)
    # user_feature_columns.append(marital_status_type_one_hot)
    # user_feature_layer_inputs[ds.MARITAL_STATUS_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.MARITAL_STATUS_COLUMN, dtype=tf.string)

    # have_child_type = feature_column.categorical_column_with_vocabulary_list(
    #     ds.HAVE_CHILD_COLUMN, ['Y', 'N'])

    # have_child_type_one_hot = feature_column.indicator_column(
    #     have_child_type)
    # user_feature_columns.append(have_child_type_one_hot)
    # user_feature_layer_inputs[ds.HAVE_CHILD_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.HAVE_CHILD_COLUMN, dtype=tf.string)

    # user_feature_layer = tf.keras.layers.DenseFeatures(user_feature_columns)

    # product_asset_class_type = feature_column.categorical_column_with_vocabulary_list(
    #     ds.PRODUCT_ASSET_CLASS_COLUMN, ['Equity Developed Market'])
    # product_asset_class_type_one_hot = feature_column.indicator_column(
    #     product_asset_class_type)
    # product_feature_columns.append(product_asset_class_type_one_hot)
    # product_feature_layer_inputs[ds.PRODUCT_ASSET_CLASS_COLUMN] = tf.keras.Input(
    #     shape=(1,), name=ds.PRODUCT_ASSET_CLASS_COLUMN, dtype=tf.string)

    # product_feature_layer = tf.keras.layers.DenseFeatures(
    #     product_feature_columns)

    # print(user_feature_columns)
    # print(product_feature_columns)

    # user_feature_layer_output = user_feature_layer(user_feature_layer_inputs)
    # product_feature_layer_output = product_feature_layer(
    #     product_feature_layer_inputs)
    # # train_ds = df_to_dataset(train, batch_size=batch_size)
    # # val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    # # test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # num_users, num_items = mat_train.shape

    # # contstruct the NeuMF Neural Network
    # with strategy.scope():
    #     model = get_model(all_user_features, all_product_features, num_users+1, (num_items*3)+1, latent_dim,
    #                       dense_layers, reg_layers, reg_mf)

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    print(model.summary())

    # # organise training input<User,Product><Label>
    # modified_x_train = []
    # for ui in user_input_train:
    #     user_df = df[df[ds.USER_INDEX_COLUMN] == ui]
    #     modified_x_train_element = []
    #     modified_x_train_element.append(
    #         user_df[ds.AGE_COLUMN].to_numpy()[0])
    #     modified_x_train_element.append(
    #         user_df[ds.GENDER_COLUMN].to_numpy()[0])
    #     modified_x_train_element.append(
    #         user_df[ds.MARITAL_STATUS_COLUMN].to_numpy()[0])
    #     modified_x_train_element.append(
    #         user_df[ds.HAVE_CHILD_COLUMN].to_numpy()[0])
    #     modified_x_train_element.append(
    #         user_df[ds.EDUCATION_COLUMN].to_numpy()[0])
    #     modified_x_train.append(modified_x_train_element)

    # modified_y_train = []
    # for ii in item_input_train:
    #     product_df = df[df[ds.PRODUCT_INDEX_COLUMN] == ii]
    #     modified_y_train_element = []
    #     modified_y_train_element.append(
    #         product_df[ds.PRODUCT_3_YR_RETURN_COLUMN].to_numpy()[0])
    #     modified_y_train_element.append(
    #         product_df[ds.PRODUCT_STD_DEV_COLUMN].to_numpy()[0])
    #     modified_y_train_element.append(
    #         product_df[ds.PRODUCT_DEVIDEND_COLUMN].to_numpy()[0])
    #     modified_y_train_element.append(
    #         product_df[ds.PRODUCT_ASSET_CLASS_COLUMN].to_numpy()[0])
    #     modified_y_train_element.append(1)
    #     modified_y_train.append(modified_y_train_element)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=Config.getNNCheckpoint(),
                                                     save_weights_only=True,
                                                     verbose=0)

    # hist = model.fit([np.array(modified_x_train), np.array(modified_y_train)], np.array(labels_train),
    #                  batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True, callbacks=[cp_callback])

    model.fit(train_ds, validation_data=val_ds,
              batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True, callbacks=[cp_callback])

    loss, accuracy = model.evaluate(test_ds)

    print("Accuracy", accuracy)
    model.save(Config.getNNModel())


if __name__ == '__main__':
    main()
