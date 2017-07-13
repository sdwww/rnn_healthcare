from keras.layers import Input, Dense, concatenate, GRU, Dropout
from keras.models import Model

from keras.utils.vis_utils import plot_model
import numpy as np

import CreateDataset


def change_to_one_zero(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if dataset[i][j] != 0:
                dataset[i][j] = 1
    return dataset


def load_train_data(month):
    dataset_jbbm_train = np.load('./data_npz/dataset_jbbm_train.npz')["arr_0"][:, -1, :] \
        .reshape(int(CreateDataset.patient_num * 0.85), CreateDataset.jbbm_num)
    dataset_drug_train = np.load('./data_npz/dataset_drug_nocost_train.npz')["arr_0"][:, -1, :] \
        .reshape(int(CreateDataset.patient_num * 0.85), CreateDataset.drug_num)
    label_jbbm_train = np.load('./data_npz/label_jbbm_train_' + str(month) + 'month.npz')["arr_0"]
    # label_drug_train = np.load('./data_npz/label_drug_nocost_train_' + str(month) + 'month.npz')["arr_0"]
    label_sick_train = np.load('./data_npz/label_sick_train_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_sick_train


def train_lr(month=3, max_epochs=10, batch_size=100):
    jbbm_input = Input(shape=(CreateDataset.jbbm_num,), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.drug_num,), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(code_input)
    sick_output = Dense(1, activation='sigmoid', name='sick_output')(code_input)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[jbbm_output, sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'sick_output': 'binary_crossentropy', 'jbbm_output': 'binary_crossentropy'},
                  loss_weights={'sick_output': 1, 'jbbm_output': 100})

    # 模型可视化
    plot_model(model, to_file='./data_png/lr_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_sick_train = load_train_data(
        month)
    model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_jbbm_train, label_sick_train],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/lr_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')


def train_mlp(month=3, max_epochs=10, batch_size=100):
    jbbm_input = Input(shape=(CreateDataset.jbbm_num,), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.drug_num,), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    hidden_layer = Dense(300)(code_input)
    jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(hidden_layer)
    sick_output = Dense(1, activation='sigmoid', name='sick_output')(hidden_layer)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[jbbm_output, sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'sick_output': 'binary_crossentropy', 'jbbm_output': 'binary_crossentropy'},
                  loss_weights={'sick_output': 1, 'jbbm_output': 100})

    # 模型可视化
    plot_model(model, to_file='./data_png/mlp_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_sick_train = load_train_data(
        month)
    model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_jbbm_train, label_sick_train],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/mlp_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')
