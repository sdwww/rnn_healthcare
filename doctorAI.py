from keras.layers import Input, Dense, concatenate, GRU, Dropout
from keras.models import Model, load_model

from keras.utils.vis_utils import plot_model
import numpy as np


def load_train_data():
    dataset_x_train = np.load('./data_npz/dataset_x_train.npz')["arr_0"]
    dataset_t_train = np.load('./data_npz/dataset_t_train.npz')["arr_0"]
    label_y_train = np.load('./data_npz/label_y_train.npz')["arr_0"]
    label_t_train = np.load('./data_npz/label_t_train.npz')["arr_0"]
    return dataset_x_train, dataset_t_train, label_y_train, label_t_train


def load_test_data():
    dataset_x_test = np.load('./data_npz/dataset_x_test.npz')["arr_0"]
    dataset_t_test = np.load('./data_npz/dataset_t_test.npz')["arr_0"]
    label_y_test = np.load('./data_npz/label_y_test.npz')["arr_0"]
    label_t_test = np.load('./data_npz/label_t_test.npz')["arr_0"]
    return dataset_x_test, dataset_t_test, label_y_test, label_t_test


def recallTop(y_true, y_pred, rank=None):
    if rank is None:
        rank = [5, 10, 15,]
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        codes = np.argsort(y_true[i])
        tops = np.argsort(y_pred[i])
        for rk in rank:
            thisOne.append(len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) * 1.0 / count)
        recall.append(thisOne)

    return (np.array(recall)).mean(axis=0).tolist()


def calculate_r_squared(trueVec, predVec):
    trueVec = np.array(trueVec)
    predVec = np.array(predVec)
    mean_duration = np.mean(trueVec)

    numerator = ((trueVec - predVec) ** 2).sum()
    denominator = ((trueVec - mean_duration) ** 2).sum()
    return 1.0 - (numerator / denominator)


def doctorAI_train(
        seqFile='./data_pkl/seqFile',
        inputDimSize=8000,
        labelFile='./data_pkl/labelFile',
        numClass=2,
        outFile='./data_pkl/outFile',
        timeFile='',
        predictTime=False,
        tradeoff=1.0,
        useLogTime=True,
        embFile='',
        embSize=200,
        embFineTune=True,
        hiddenDimSize=[200, 200],
        batchSize=100,
        max_epochs=10,
        L2_output=0.001,
        L2_time=0.001,
        dropout_rate=0.5,
        logEps=1e-8,
        verbose=False):
    x_input = Input(shape=(96, 2094), name='code_input')
    embed = Dense(units=500, activation='tanh')(x_input)
    d_input = Input(shape=(96, 1), name='time_input')
    embed_d = concatenate([embed, d_input])
    gru_1 = GRU(units=300, return_sequences=True)(embed_d)
    dropout_1 = Dropout(rate=0.2)(gru_1)
    gru_2 = GRU(units=300, return_sequences=False)(dropout_1)
    dropout_2 = Dropout(rate=0.2)(gru_2)
    x_output = Dense(301, activation='softmax', name='code_output')(dropout_2)
    d_output = Dense(1, activation='relu', name='time_output')(dropout_2)
    model = Model(inputs=[x_input, d_input], outputs=[x_output, d_output])
    model.compile(optimizer='rmsprop',
                  loss={'code_output': 'binary_crossentropy', 'time_output': 'mean_squared_error'},
                  loss_weights={'code_output': 10.0, 'time_output': 0.001})

    # # 模型可视化
    # plot_model(model, to_file='./data_png/model.png', show_shapes=True)

    dataset_x_train, dataset_t_train, label_y_train, label_t_train = load_train_data()
    model.fit(x=[dataset_x_train, dataset_t_train], y=[label_y_train, label_t_train],
              epochs=max_epochs, batch_size=batchSize, validation_split=0.2)
    model.save('./data_h5/model_'+str(max_epochs)+'epochs.h5')


def doctorAI_test(filename):
    dataset_x_test, dataset_t_test, label_y_test, label_t_test = load_test_data()
    model = load_model(filename)
    pred_y_test, pred_t_test = model.predict(x=[dataset_x_test, dataset_t_test])
    print(pred_t_test[3],label_t_test[3])
    print("top5,top10,top15 recall分别为",recallTop(label_y_test, pred_y_test))
    print(calculate_r_squared(label_t_test, pred_t_test))
