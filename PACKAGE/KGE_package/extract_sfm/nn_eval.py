from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from train import PatTypeOH

import pickle
import numpy as np
from config import *
import sys
np.set_printoptions(threshold=sys.maxsize)


if __name__ == "__main__":
    model = keras.models.load_model("model.h5", custom_objects={'PatTypeOH': PatTypeOH})
    print(model.summary())

    pp_filename = "data_XY.p"
    X_data, Y_data = pickle.load(open(pp_filename, "rb"))

    print(X_data.shape)
    print(Y_data.shape)
    # print(pattern_1hot_len)

    X_train, X_test = X_data[:TRAIN_SPLIT], X_data[TRAIN_SPLIT:]
    Y_train, Y_test = Y_data[:TRAIN_SPLIT], Y_data[TRAIN_SPLIT:]

    # print(np.sum(Y_data[np.sum(X_data[:, :, 0], axis = 1) == 0]))
    # print(np.sum(np.sum(X_data[:, :, 0], axis = 1) == 0))
    # print(X_data.shape)
    # print(X_data[:, :, 0].shape)
    #
    # print(Y_data.shape[0] - np.sum(Y_data))
    # print(X_data[np.logical_and(np.sum(X_data[:, :, 0], axis = 1) != 0, np.sum(Y_data, axis = 1) == 0)][:, :, 0])

    Y_pred = model.predict_classes(X_test)
    Y_1hot = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis = -1)
    cout_correct = 0
    count_zero = 0
    count_nonzero = 0
    for idx in range(X_test.shape[0]):
        # print(Y_true[idx], Y_pred[idx])
        if Y_pred[idx] == Y_true[idx]:
            cout_correct += 1
            if np.sum(Y_test[idx]) == 0:
                count_zero += 1
            else:
                count_nonzero += 1
    # print(count_zero, count_nonzero)

    # print(Y_test[np.sum(Y_test, axis = 1) == 0])
    print("Number of zero vector outputs: ", np.sum(np.sum(Y_test, axis = 1) == 0), "/", Y_test.shape[0])
    print("Accuracy: ", cout_correct / Y_pred.shape[0])
    print(np.sum(Y_test, axis = 0))
    print(np.sum(Y_pred == 0))
    print(np.sum(Y_pred == 1))
    print(np.sum(Y_pred == 2))
