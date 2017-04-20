from keras.layers import Input, Dense, concatenate, GRU, Dropout, LSTM
from keras.models import Model
from keras.callbacks import History, EarlyStopping, TensorBoard

from keras.utils.vis_utils import plot_model
import numpy as np
import CreateDataset
import FileOptions


def load_train_data(month):
    dataset_jbbm_train = np.load('./data_npz/dataset_jbbm_train.npz')["arr_0"]
    dataset_drug_train = np.load('./data_npz/dataset_drug_nocost_train.npz')["arr_0"]
    label_jbbm_train = np.load('./data_npz/label_jbbm_train_' + str(month) + 'month.npz')["arr_0"]
    label_drug_train = np.load('./data_npz/label_drug_nocost_train_' + str(month) + 'month.npz')["arr_0"]
    label_sick_train = np.load('./data_npz/label_sick_train_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train


def train_model(
        emb_size=500,
        hidden_size=None,
        batch_size=100,
        max_epochs=10,
        rnn_unit='gru',
        dropout_rate=0.2,
        month=3):
    jbbm_input = Input(shape=(CreateDataset.visit_num, CreateDataset.jbbm_num), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.visit_num, CreateDataset.drug_num), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    embed_layer = Dense(units=emb_size, activation='tanh')(code_input)
    if len(hidden_size) == 2 and rnn_unit == 'gru':
        gru_1 = GRU(units=hidden_size[0], return_sequences=True)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(gru_1)
        gru_2 = GRU(units=hidden_size[1], return_sequences=False)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(gru_2)
        jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(dropout_2)
        # drug_output = Dense(CreateDataset.drug_categ_num, activation='softmax', name='drug_output')(dropout_2)
        sick_output = Dense(1, activation='sigmoid', name='sick_output')(dropout_2)
    elif len(hidden_size) == 1 and rnn_unit == 'gru':
        gru_1 = GRU(units=hidden_size[0], return_sequences=False)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(gru_1)
        jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(dropout_1)
        # drug_output = Dense(CreateDataset.drug_categ_num, activation='softmax', name='drug_output')(dropout_1)
        sick_output = Dense(1, activation='sigmoid', name='sick_output')(dropout_1)
    elif len(hidden_size) == 2 and rnn_unit == 'lstm':
        lstm_1 = LSTM(units=hidden_size[0], return_sequences=True)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        lstm_2 = LSTM(units=hidden_size[1], return_sequences=False)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(lstm_2)
        jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(dropout_2)
        # drug_output = Dense(CreateDataset.drug_categ_num, activation='softmax', name='drug_output')(dropout_2)
        sick_output = Dense(1, activation='sigmoid', name='sick_output')(dropout_2)
    else:
        lstm_1 = LSTM(units=hidden_size[0], return_sequences=False)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        jbbm_output = Dense(CreateDataset.jbbm_categ_num, activation='softmax', name='jbbm_output')(dropout_1)
        # drug_output = Dense(CreateDataset.drug_categ_num, activation='softmax', name='drug_output')(dropout_1)
        sick_output = Dense(1, activation='sigmoid', name='sick_output')(dropout_1)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[jbbm_output, sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'jbbm_output': 'binary_crossentropy',
                        # 'drug_output': 'binary_crossentropy',
                        'sick_output': 'binary_crossentropy'},
                  loss_weights={'jbbm_output': 50, 'sick_output': 1})

    # 模型可视化
    plot_model(model, to_file='./data_png/rnn_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train = load_train_data(
        month)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    tensor_board = TensorBoard(log_dir='./tensor_log')
    hist = model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_jbbm_train, label_sick_train],
                     epochs=max_epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop,tensor_board])
    print(len(hist.history['loss']))
    file_name = 'rnn' + str(len(hidden_size)) + '_' + str(emb_size) + 'emb_' + str(hidden_size) \
                + 'hidden_' + '_' + rnn_unit + '_' + str(20) + 'epochs_' + str(month) + 'month'
    model.save('./data_h5/' + file_name + '.h5')
    FileOptions.dump_pkl(hist.history, '/data_history/' + file_name + '.history')
