import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle
import numpy as np
from config import *
import sys
# np.set_printoptions(threshold=sys.maxsize)

class PatTypeOH(layers.Layer):
    def __init__(self, pat_1hot_len, type_1hot_len, hidden_pat = 10, hidden_type = 6, **kwargs):
        self.hidden_pat = hidden_pat
        self.hidden_type = hidden_type
        self.pat_1hot_len = pat_1hot_len
        self.type_1hot_len = type_1hot_len
        super(PatTypeOH, self).__init__(**kwargs)


    def build(self, input_shape):
        self.w_pat = self.add_weight(shape=(self.pat_1hot_len, self.hidden_pat),
                                 initializer='random_normal',
                                 trainable=True)
        self.w_type = self.add_weight(shape=(self.type_1hot_len, self.hidden_type),
                                 initializer='random_normal',
                                 trainable=True)
        super(PatTypeOH, self).build(input_shape)

    def call(self, inputs):
        input_len = inputs.shape[1]
        pat_num = (input_len - self.w_type.shape[0]) // self.w_pat.shape[0]
        w_pat_x_list = []
        for pat_idx in range(pat_num):
            cur_x_start = self.w_pat.shape[0] * pat_idx
            cur_x_end = self.w_pat.shape[0] * (pat_idx + 1)
            w_pat_x_list.append(tf.matmul(inputs[:, cur_x_start:cur_x_end], self.w_pat))
        w2x = tf.matmul(inputs[:, self.w_pat.shape[0] * pat_num:], self.w_type)
        w_pat_x_list.append(w2x)
        return tf.concat(w_pat_x_list, 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_pat': self.hidden_pat,
            'hidden_type': self.hidden_type,
            'pat_1hot_len': self.pat_1hot_len,
            'type_1hot_len': self.type_1hot_len,
        })
        return config

if __name__ == "__main__":
    X_data, Y_data = pickle.load(open(dataset_filename, "rb"))
    all_patterns = pickle.load(open(patterns_filename, "rb"))
    pattern_1hot_len = len(all_patterns)

    print(X_data.shape)
    print(Y_data.shape)
    print(pattern_1hot_len)

    X_train, X_test = X_data[:TRAIN_SPLIT], X_data[TRAIN_SPLIT:]
    Y_train, Y_test = Y_data[:TRAIN_SPLIT], Y_data[TRAIN_SPLIT:]

    if LOAD_MODEL:
        model = Sequential([
            PatTypeOH((pattern_1hot_len + 1), len(ALL_NE_TYPES), 6, 3),
            Activation('relu'),
            # Flatten(),
            # Dense(PER_NUM * 2, activation = 'relu'),
            # Dense(PER_NUM * 2, activation = 'relu'),
            # Dense(PER_NUM * 2, activation = 'relu'),
            Dense(PER_NUM, activation = 'softmax'),
        ])

        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    else:
        model = keras.models.load_model(model_path, custom_objects={'PatTypeOH': PatTypeOH})

    checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=1)
    model.fit(X_train, Y_train,
                epochs=EPOCH_NUM,
                batch_size=32,
                validation_data=(X_test, Y_test),
                shuffle=True,
                callbacks=[checkpoint])

    model.save(model_path)
