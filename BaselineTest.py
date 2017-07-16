import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import DatasetProcess


def load_data(month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease \
        = DatasetProcess.load_test_data(month)
    test_disease = test_disease[:, -1, :].reshape(int(DatasetProcess.patient_num * DatasetProcess.test_ratio) + 1,
                                                  DatasetProcess.disease_num)
    test_drug = test_drug[:, -1, :].reshape(int(DatasetProcess.patient_num * DatasetProcess.test_ratio) + 1,
                                            DatasetProcess.drug_num)
    return test_info, test_disease, test_drug, test_label_probability, test_label_disease


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
    predict_vec = predict_vec // (0.5 + 1e-3)
    recall = recall_score(true_vec, predict_vec)
    return recall


def calculate_accuracy(true_vec, predict_vec):
    predict_vec = predict_vec // (0.5 + 1e-3)
    recall = accuracy_score(true_vec, predict_vec)
    return recall


def calculate_precision(true_vec, predict_vec):
    predict_vec = predict_vec // (0.5 + 1e-3)
    precision = precision_score(true_vec, predict_vec)
    return precision


def calculate_f1score(true_vec, predict_vec):
    predict_vec = predict_vec // (0.5 + 1e-3)
    f_score = f1_score(true_vec, predict_vec)
    return f_score


def test_probability(filename, month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease = load_data(month)
    model = load_model('./data_h5/' + filename)
    predict_probability_test, predict_disease_test, = model.predict(x=[test_disease, test_drug])
    auc = calculate_auc(test_label_probability, predict_probability_test)
    acc = calculate_accuracy(test_label_probability, predict_probability_test)
    precision = calculate_precision(test_label_probability, predict_probability_test)
    recall = calculate_recall(test_label_probability, predict_probability_test)
    f_score = calculate_f1score(test_label_probability, predict_probability_test)
    return [auc, acc, precision, recall, f_score]


def test_disease(filename, month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease = load_data(month)
    model = load_model('./data_h5/' + filename)
    predict_probability_test, predict_disease_test = model.predict(x=[test_disease, test_drug])
    top1_pre, top2_pre, top3_pre = precision_top(test_label_disease, predict_disease_test)
    return [top1_pre, top2_pre, top3_pre]


def test_last_time_jbbm(month):
    test_info, test_disease, test_drug, test_label_probability, test_label_disease = load_data(month)
    top1_pre, top2_pre, top3_pre = precision_top(test_label_disease, test_disease)
    return [top1_pre, top2_pre, top3_pre]


if __name__ == "__main__":
    month_list = [3, 6, 9, 12]
    # 测试逻辑回归模型
    for i in month_list:
        print(test_probability('lr_10epochs_' + str(i) + 'month.h5', month=i))
        print(test_disease('lr_10epochs_' + str(i) + 'month.h5', month=i))

    # 测试多层感知机模型
    for i in month_list:
        print(test_probability('mlp_10epochs_' + str(i) + 'month.h5', month=i))
        print(test_disease('mlp_10epochs_' + str(i) + 'month.h5', month=i))

    # 测试最后一次
    for i in month_list:
        print(test_last_time_jbbm(month=i))
