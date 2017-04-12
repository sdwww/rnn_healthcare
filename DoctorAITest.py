import numpy as np
from keras.models import Model, load_model


def load_test_data():
    dataset_x_test = np.load('./data_npz/dataset_x_test.npz')["arr_0"]
    dataset_t_test = np.load('./data_npz/dataset_t_test.npz')["arr_0"]
    label_y_test = np.load('./data_npz/label_y_test.npz')["arr_0"]
    label_t_test = np.load('./data_npz/label_t_test.npz')["arr_0"]
    return dataset_x_test, dataset_t_test, label_y_test, label_t_test


def recallTop(y_true, y_pred, rank=None):
    if rank is None:
        rank = [5, 10, 15, ]
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


def doctorAI_test(filename):
    dataset_x_test, dataset_t_test, label_y_test, label_t_test = load_test_data()
    model = load_model(filename)
    pred_y_test, pred_t_test = model.predict(x=[dataset_x_test, dataset_t_test])
    print(pred_t_test[3], label_t_test[3])
    print("top5,top10,top15 recall分别为", recallTop(label_y_test, pred_y_test))
    print(calculate_r_squared(label_t_test, pred_t_test))
