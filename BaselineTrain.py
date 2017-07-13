from keras.layers import Input, Dense, concatenate
from keras.models import Model

import DatasetProcess


def load_data(month):
    train_info, train_disease, train_drug, train_label_probability, train_label_disease \
        = DatasetProcess.load_train_data(month)
    train_disease = train_disease[:, -1, :].reshape(int(DatasetProcess.patient_num * (1 - DatasetProcess.test_ratio)),
                                                    DatasetProcess.disease_num)
    train_drug = train_drug[:, -1, :].reshape(int(DatasetProcess.patient_num * (1 - DatasetProcess.test_ratio)),
                                              DatasetProcess.drug_num)
    return train_info, train_disease, train_drug, train_label_probability, train_label_disease


def train_lr(month=3, max_epochs=10, batch_size=100):
    disease_input = Input(shape=(DatasetProcess.disease_num,), name='disease_input')
    drug_input = Input(shape=(DatasetProcess.drug_num,), name='drug_input')
    code_input = concatenate([disease_input, drug_input])
    disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                           name='disease_output')(code_input)
    probability_output = Dense(1, activation='sigmoid', name='probability_output')(code_input)
    model = Model(inputs=[disease_input, drug_input], outputs=[probability_output, disease_output])
    model.compile(optimizer='rmsprop',
                  loss={'probability_output': 'binary_crossentropy', 'disease_output': 'binary_crossentropy'},
                  loss_weights={'probability_output': 1, 'disease_output': 100})
    train_info, train_disease, train_drug, train_label_probability, train_label_disease, = load_data(month)
    model.fit(x=[train_disease, train_drug], y=[train_label_probability, train_label_disease],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/lr_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')


def train_mlp(month=3, max_epochs=10, batch_size=100):
    disease_input = Input(shape=(DatasetProcess.disease_num,), name='disease_input')
    drug_input = Input(shape=(DatasetProcess.drug_num,), name='drug_input')
    code_input = concatenate([disease_input, drug_input])
    hidden_layer = Dense(300)(code_input)
    disease_output = Dense(DatasetProcess.disease_category_num, activation='softmax',
                           name='disease_output')(hidden_layer)
    probability_output = Dense(1, activation='sigmoid', name='probability_output')(hidden_layer)
    model = Model(inputs=[disease_input, drug_input], outputs=[probability_output, disease_output])
    model.compile(optimizer='rmsprop',
                  loss={'probability_output': 'binary_crossentropy', 'disease_output': 'binary_crossentropy'},
                  loss_weights={'probability_output': 1, 'disease_output': 100})
    train_info, train_disease, train_drug, train_label_probability, train_label_disease = load_data(month)
    model.fit(x=[train_disease, train_drug], y=[train_label_probability, train_label_disease],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/mlp_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')


if __name__ == "__main__":

    # 训练逻辑回归模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        train_lr(month=i, max_epochs=10, batch_size=100)

    # 训练多层感知机模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        train_mlp(month=i, max_epochs=10, batch_size=64)
