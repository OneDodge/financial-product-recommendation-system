from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

#!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL, nrows=10000)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Define method to create tf.data dataset from Pandas Dataframe
# This worked with tf 2.0 but does not work with tf 2.2


def df_to_dataset_tf_2_0(dataframe, label_column, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    #labels = dataframe.pop(label_column)
    labels = dataframe[label_column]

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def df_to_dataset(dataframe, label_column, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_column)
    #labels = dataframe[label_column]

    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe.to_dict(orient='list'), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, label_column='target', batch_size=batch_size)
val_ds = df_to_dataset(val, label_column='target',
                       shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, label_column='target',
                        shuffle=False, batch_size=batch_size)

age = feature_column.numeric_column("age")

feature_columns = []
feature_layer_inputs = {}

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))
    feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)

# bucketized cols
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)
feature_layer_inputs['thal'] = tf.keras.Input(
    shape=(1,), name='thal', dtype=tf.string)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column(
    [age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
feature_layer_outputs = feature_layer(feature_layer_inputs)

x = layers.Dense(128, activation='relu')(feature_layer_outputs)
x = layers.Dense(64, activation='relu')(x)

baggage_pred = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(
    inputs=[v for v in feature_layer_inputs.values()], outputs=baggage_pred)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds)
