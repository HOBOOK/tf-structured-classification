import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

URL = 'https://vridge-spc.vazil.me/api/storage/file/download/MmYwZTJkYzk0MzNjNDZiMWIyMWY4MWZkYzMyNThmZTYvc2VyaWFsL3AtNDEwMDliNmFjZmJmNGViZWFkZTUwYTZhMjBjZjlkMWUvMjA5NGUwNjBiMWQ0NGUxZWEyMTFlODk4N2JhMzEyMDQuY3N2'
#URL = 'C:/dev/tf-structured-classification/spc_sample_v1.csv'
#dataframe = pd.read_csv(URL, sep='\t', engine='python', encoding='cp949')
dataframe = pd.read_csv(URL)
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), '훈련 샘플')
print(len(val), '검증 샘플')
print(len(test), '테스트 샘플')

  # 판다스 데이터프레임으로부터 tf.data 데이터셋을 만들기 위한 함수
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('lot')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

# batch_size = 5 # 예제를 위해 작은 배치 크기를 사용합니다.
# train_ds = df_to_dataset(train, batch_size=batch_size)
# val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# for feature_batch, label_batch in train_ds.take(1):
#   print('전체 특성:', list(feature_batch.keys()))
#   print('나이 특성의 배치:', feature_batch['data'])
#   print('타깃의 배치:', label_batch )


# # 특성 열을 시험해 보기 위해 샘플 배치를 만듭니다.
# example_batch = next(iter(train_ds))[0]

# # 특성 열을 만들고 배치 데이터를 변환하는 함수
# def demo(feature_column):
#   feature_layer = layers.DenseFeatures(feature_column)
#   print(feature_layer(example_batch).numpy())

# d = feature_column.numeric_column("data")
# demo(d)


feature_columns = []

# 수치형 열
for header in ['data']:
  feature_columns.append(feature_column.numeric_column(header))

print(f'feature_columns:: {feature_columns}')

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

print(f'feature_layer :: {feature_layer}')

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("정확도", accuracy)