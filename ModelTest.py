import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import DatasetProcess


def recall_top(y_true, y_predict, rank=None):
    if rank is None:
        rank = [1, 2, 3]
    recall = list()
    for i in range(len(y_predict)):
        this_one = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_predict[i])
            for rk in rank:
                this_one.append(
                    len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) * 1.0 / count)
            recall.append(this_one)

    return (np.array(recall)).mean(axis=0).tolist()


def precision_top(y_true, y_predict, rank=None):
    if rank is None:
        rank = [1, 2, 3]
    pre = list()
    for i in range(len(y_predict)):
        this_one = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_predict[i])
            for rk in rank:
                if len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) >= 1:
                    this_one.append(1)
                else:
                    this_one.append(0)
            pre.append(this_one)

    return (np.array(pre)).mean(axis=0).tolist()


def calculate_r_squared(true_vec, predict_vec):
    true_vec = np.array(true_vec)
    predict_vec = np.array(predict_vec)
    mean_duration = np.mean(true_vec)

    numerator = ((true_vec - predict_vec) ** 2).sum()
    denominator = ((true_vec - mean_duration) ** 2).sum()
    return 1.0 - (numerator / denominator)


def calculate_auc(true_vec, predict_vec):
    auc = roc_auc_score(true_vec, predict_vec)
    return auc


def calculate_recall(true_vec, predict_vec):
    predict_vec = predict_vec // 0.5
    recall = recall_score(true_vec, predict_vec)
    return recall


def calculate_accuracy(true_vec, predict_vec):
    predict_vec = predict_vec // 0.5
    recall = accuracy_score(true_vec, predict_vec)
    return recall


def calculate_precision(true_vec, predict_vec):
    predict_vec = predict_vec // 0.5
    precision = precision_score(true_vec, predict_vec)
    return precision


def calculate_f1score(true_vec, predict_vec):
    predict_vec = predict_vec // 0.5
    f_score = f1_score(true_vec, predict_vec)
    return f_score


def test_model_probability(filename, month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease \
        = DatasetProcess.load_test_data(month)
    model = load_model('./data_h5/' + filename)
    predict_disease_test, predict_probability_test = model.predict(x=[test_disease, test_drug])
    auc = calculate_auc(test_label_probability, predict_probability_test)
    acc = calculate_accuracy(test_label_probability, predict_probability_test)
    precision = calculate_precision(test_label_probability, predict_probability_test)
    recall = calculate_recall(test_label_probability, predict_probability_test)
    f_score = calculate_f1score(test_label_probability, predict_probability_test)
    return [auc, acc, precision, recall, f_score]


def test_model_disease(filename, month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease \
        = DatasetProcess.load_test_data(month)
    model = load_model('./data_h5/' + filename)
    predict_probability_test, predict_disease_test = model.predict(x=[test_disease, test_drug])
    top1_pre, top2_pre, top3_pre = precision_top(test_label_disease, predict_disease_test)
    return [top1_pre, top2_pre, top3_pre]


if __name__ == "__main__":
    # 测试RNN模型
    month_list = [3, 6, 9, 12]
    emb_list = [500, 1000]
    hidden_list = [[300, 300], [300]]
    rnn_unit_list = ['gru', 'lstm']
    for month in month_list:
        for emb in emb_list:
            for hidden in hidden_list:
                for rnn_unit in rnn_unit_list:
                    file_name = 'rnn' + str(len(hidden)) + '_' + str(emb) + 'emb_' + str(hidden) \
                                + 'hidden_' + '_' + rnn_unit + '_' + str(20) + 'epochs_' + str(
                        month) + 'month'
                    print(file_name)
                    test_model_disease(file_name + '.h5', month=month)
