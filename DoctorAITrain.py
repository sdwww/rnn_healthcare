from keras.layers import Input, Dense, concatenate, GRU, Dropout
from keras.models import Model

from keras.utils.vis_utils import plot_model
import numpy as np
import CreateDataset


def load_train_data(month):
    dataset_jbbm_train = np.load('./data_npz/dataset_jbbm_train.npz')["arr_0"]
    dataset_drug_train = np.load('./data_npz/dataset_drug_nocost_train.npz')["arr_0"]
    label_jbbm_train = np.load('./data_npz/label_jbbm_train_' + str(month) + 'month.npz')["arr_0"]
    label_drug_train = np.load('./data_npz/label_drug_nocost_train_' + str(month) + 'month.npz')["arr_0"]
    label_sick_train = np.load('./data_npz/label_sick_train_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train


def train_model(
        emb_size=500,
        hidden_size=[200, 200],
        batch_size=100,
        max_epochs=10,
        L2_output=0.001,
        dropout_rate=0.2,
        month=3):
    jbbm_input = Input(shape=(CreateDataset.visit_num, CreateDataset.jbbm_num), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.visit_num, CreateDataset.drug_num), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    embed_layer = Dense(units=emb_size, activation='tanh')(code_input)
    gru_1 = GRU(units=hidden_size[0], return_sequences=True)(embed_layer)
    dropout_1 = Dropout(rate=dropout_rate)(gru_1)
    gru_2 = GRU(units=hidden_size[1], return_sequences=False)(dropout_1)
    dropout_2 = Dropout(rate=dropout_rate)(gru_2)
    jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(dropout_2)
    drug_output = Dense(CreateDataset.drug_categ_num, activation='softmax', name='drug_output')(dropout_2)
    sick_output = Dense(1, activation='sigmoid', name='sick_output')(dropout_2)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[jbbm_output, drug_output, sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'jbbm_output': 'binary_crossentropy', 'drug_output': 'binary_crossentropy',
                        'sick_output': 'binary_crossentropy'},
                  loss_weights={'jbbm_output': 1, 'drug_output': 1, 'sick_output': 1},metrics=[''])

    # 模型可视化
    plot_model(model, to_file='./data_png/rnn_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train = load_train_data(
        month)
    model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_jbbm_train, label_drug_train, label_sick_train],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/model_' + str(max_epochs) + 'epochs_'+str(month)+'month.h5')
