import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dense, concatenate, GRU, Dropout, LSTM, SimpleRNN
from keras.models import Model

import DatasetProcess


def train_model(
        emb_size=500,
        hidden_size=None,
        batch_size=100,
        max_epochs=10,
        rnn_unit='gru',
        dropout_rate=0.2,
        month=3):
    disease_input = Input(shape=(DatasetProcess.visit_num, DatasetProcess.disease_num), name='disease_input')
    drug_input = Input(shape=(DatasetProcess.visit_num, DatasetProcess.drug_num), name='drug_input')
    code_input = concatenate([disease_input, drug_input])
    embed_layer = Dense(units=emb_size, activation='tanh')(code_input)
    if len(hidden_size) == 2 and rnn_unit == 'gru':
        gru_1 = GRU(units=hidden_size[0], return_sequences=True)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(gru_1)
        gru_2 = GRU(units=hidden_size[1], return_sequences=False)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(gru_2)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                               name='disease_output')(dropout_2)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_2)
    elif len(hidden_size) == 1 and rnn_unit == 'gru':
        gru_1 = GRU(units=hidden_size[0], return_sequences=False)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(gru_1)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                               name='disease_output')(dropout_1)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_1)
    elif len(hidden_size) == 2 and rnn_unit == 'lstm':
        lstm_1 = LSTM(units=hidden_size[0], return_sequences=True)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        lstm_2 = LSTM(units=hidden_size[1], return_sequences=False)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(lstm_2)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                               name='disease_output')(dropout_2)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_2)
    elif len(hidden_size) == 1 and rnn_unit == 'lstm':
        lstm_1 = LSTM(units=hidden_size[0], return_sequences=False)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                               name='disease_output')(dropout_1)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_1)
    elif len(hidden_size) == 2 and rnn_unit == 'simplernn':
        lstm_1 = SimpleRNN(units=hidden_size[0], return_sequences=True)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        lstm_2 = SimpleRNN(units=hidden_size[1], return_sequences=False)(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(lstm_2)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax', name='disease_output')(
            dropout_2)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_2)
    else:
        lstm_1 = SimpleRNN(units=hidden_size[0], return_sequences=False)(embed_layer)
        dropout_1 = Dropout(rate=dropout_rate)(lstm_1)
        disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax', name='disease_output')(
            dropout_1)
        probability_output = Dense(1, activation='sigmoid', name='probability_output')(dropout_1)
    model = Model(inputs=[disease_input, drug_input], outputs=[probability_output, disease_output])
    model.compile(optimizer='rmsprop',
                  loss={'disease_output': 'binary_crossentropy',
                        'probability_output': 'binary_crossentropy'},
                  loss_weights={'probability_output': 1, 'disease_output': 100})
    train_info, train_disease, train_drug, train_label_probability, train_label_disease \
        = DatasetProcess.load_train_data(month)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    tensor_board = TensorBoard(log_dir='./tensor_log')
    model.fit(x=[train_disease, train_drug], y=[train_label_probability, train_label_disease],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2,
              callbacks=[early_stop, tensor_board])
    file_name = 'rnn' + str(len(hidden_size)) + '_' + str(emb_size) + 'emb_' + str(hidden_size) \
                + 'hidden_' + '_' + rnn_unit + '_' + str(20) + 'epochs_' + str(month) + 'month'
    model.save('./data_h5/' + file_name + '.h5')


if __name__ == "__main__":
    # 训练RNN模型
    month_list = [3, 6, 9, 12]
    emb_list = [500, 1000]
    hidden_list = [[300, 300], [300]]
    rnn_uint_list = ['simplernn']
    for month in month_list:
        for emb in emb_list:
            for hidden in hidden_list:
                for rnn_unit in rnn_uint_list:
                    train_model(month=month, max_epochs=100,
                                batch_size=100, emb_size=emb,
                                hidden_size=hidden, rnn_unit=rnn_unit)
