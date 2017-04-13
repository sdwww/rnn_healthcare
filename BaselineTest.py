import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import BaselineTrain


def load_test_data(month):
    dataset_jbbm_test = BaselineTrain.change_to_one_zero(np.sum(
        np.load('./data_npz/dataset_jbbm_test.npz')["arr_0"], axis=1))
    dataset_drug_test = BaselineTrain.change_to_one_zero(np.sum(
        np.load('./data_npz/dataset_drug_nocost_test.npz')["arr_0"], axis=1))
    label_jbbm_test = np.load('./data_npz/label_jbbm_test_' + str(month) + 'month.npz')["arr_0"]
    label_drug_test = np.load('./data_npz/label_drug_nocost_test_' + str(month) + 'month.npz')["arr_0"]
    label_sick_test = np.load('./data_npz/label_sick_test_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_drug_test, label_sick_test


def calculate_auc(true_vec, pred_vec):
    auc = roc_auc_score(true_vec, pred_vec)
    return auc


def calculate_recall(true_vec, pred_vec):
    pred_vec = pred_vec // 0.5
    recall = recall_score(true_vec, pred_vec)
    return recall


def calculate_accuracy(true_vec, pred_vec):
    pred_vec = pred_vec // 0.5
    recall = accuracy_score(true_vec, pred_vec)
    return recall


def calculate_precision(true_vec, pred_vec):
    pred_vec = pred_vec // 0.5
    precision = precision_score(true_vec, pred_vec)
    return precision


def test_lr(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_drug_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    # print('前十个人的预测情况和患病情况分别为')
    # for i in range(10):
    #     print(pred_sick_test[i], label_sick_test[i])
    # print("top5,top10,top15 recall分别为", recallTop(label_jbbm_test, pred_jbbm_test))
    # print("top5,top10,top15 recall分别为", recallTop(label_drug_test, pred_drug_test))
    auc = calculate_auc(label_sick_test, pred_sick_test)
    acc = calculate_accuracy(label_sick_test, pred_sick_test)
    precision = calculate_precision(label_sick_test, pred_sick_test)
    recall = calculate_recall(label_sick_test, pred_sick_test)
    return [auc, acc, precision, recall]


def test_mlp(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_drug_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    # print('前十个人的预测情况和患病情况分别为')
    # for i in range(10):
    #     print(pred_sick_test[i], label_sick_test[i])
    # print("top5,top10,top15 recall分别为", recallTop(label_jbbm_test, pred_jbbm_test))
    # print("top5,top10,top15 recall分别为", recallTop(label_drug_test, pred_drug_test))
    auc = calculate_auc(label_sick_test, pred_sick_test)
    acc = calculate_accuracy(label_sick_test, pred_sick_test)
    precision = calculate_precision(label_sick_test, pred_sick_test)
    recall = calculate_recall(label_sick_test, pred_sick_test)
    return [auc, acc, precision, recall]
