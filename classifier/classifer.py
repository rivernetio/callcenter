import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import DNNClassifier


CATEGORICAL_COLUMNS = []
CONTINUOUS_COLUMNS = ['MEAN_INTERVAL_CALL', 'SD_INTERVAL_CALL',
                      'NUM_CALL', 'MEAN_DURATION',
                      'MOST_DURATION', 'MOST_DURATION_NUM',
                      'SD_DURATION', 'TOTAL_DURATION',
                      'TRK_NUM',
]
LABEL_COLUMN = 'TAG'


def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}

    feature_cols = dict(continuous_cols)

    if CATEGORICAL_COLUMNS:
        categorical_cols = {
            k: tf.SparseTensor(
                indices=[[i, 0] for i in range(df[k].size)],
                values=df[k].values,
                dense_shape=[df[k].size, 1])
            for k in CATEGORICAL_COLUMNS}

        feature_cols.update(categorical_cols)
    label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])

    return feature_cols, label


def create_columns(continuous_columns):
    deep_columns = []
    for column in continuous_columns:
        column = tf.contrib.layers.real_valued_column(column)
        deep_columns.append(column)
    return deep_columns


def main():
    training_data = pd.read_csv('../data/20180105_label.csv',
                                skipinitialspace=True,
                                engine='python',
                                dtype=np.float64,
                                iterator=True,
                                )

    test_data = pd.read_csv('../data/20180107_label.csv',
                            skipinitialspace=True,
                            engine='python',
                            dtype=np.float64,
                            iterator=True,
                            )
    deep_columns = create_columns(CONTINUOUS_COLUMNS)

    model = DNNClassifier(feature_columns=deep_columns,
                      model_dir='./model',
                      hidden_units=[10, 10],
                      n_classes=2,
                      input_layer_min_slice_size=10000)

    tf.logging.set_verbosity(tf.logging.INFO)
    training_data_chunk = training_data.get_chunk(1000000000)
    model.fit(input_fn=lambda: input_fn(training_data_chunk),
          steps=100)

    tf.logging.info("end fit model")

    test_data_chunk = test_data.get_chunk(10000)

    accuracy = model.evaluate(input_fn=lambda: input_fn(test_data_chunk),
                          steps=100)['accuracy']
    print(accuracy * 100)

if __name__ == '__main__':
    main()

