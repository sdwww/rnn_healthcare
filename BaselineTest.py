import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import BaselineTrain
import CreateDataset


def load_test_data(month):
    # dataset_jbbm_test = BaselineTrain.change_to_one_zero(np.sum(
    #     np.load('./data_npz/dataset_jbbm_test.npz')["arr_0"], axis=1))
    # dataset_drug_test = BaselineTrain.change_to_one_zero(np.sum(
    #     np.load('./data_npz/dataset_drug_nocost_test.npz')["arr_0"], axis=1))
    dataset_jbbm_test = np.load('./data_npz/dataset_jbbm_test.npz')["arr_0"][:, -1, :] \
        .reshape(int(CreateDataset.patient_num * 0.15 + 1), CreateDataset.jbbm_num)
    dataset_drug_test = np.load('./data_npz/dataset_drug_nocost_test.npz')["arr_0"][:, -1, :] \
        .reshape(int(CreateDataset.patient_num * 0.15 + 1), CreateDataset.drug_num)
    label_jbbm_test = np.load('./data_npz/label_jbbm_test_' + str(month) + 'month.npz')["arr_0"]
    # label_drug_test = np.load('./data_npz/label_drug_nocost_test_' + str(month) + 'month.npz')["arr_0"]
    label_sick_test = np.load('./data_npz/label_sick_test_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_sick_test


def recall_top(y_true, y_pred, rank=None):
    if rank is None:
        rank = [1, 2, 3]
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_pred[i])
            for rk in rank:
                thisOne.append(
                    len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) * 1.0 / count)
            recall.append(thisOne)

    return (np.array(recall)).mean(axis=0).tolist()


def precision_top(y_true, y_pred, rank=None):
    if rank is None:
        rank = [1, 2, 3]
    pre = list()
    for i in range(len(y_pred)):
        thisOne = list()
        count = 0
        for j in y_true[i]:
            if j == 1:
                count += 1
        if count:
            codes = np.argsort(y_true[i])
            tops = np.argsort(y_pred[i])
            for rk in rank:
                if len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:])))>=1:
                    thisOne.append(1)
                else:
                    thisOne.append(0)
            pre.append(thisOne)

    return (np.array(pre)).mean(axis=0).tolist()


def calculate_r_squared(true_vec, pred_vec):
    true_vec = np.array(true_vec)
    pred_vec = np.array(pred_vec)
    mean_duration = np.mean(true_vec)

    numerator = ((true_vec - pred_vec) ** 2).sum()
    denominator = ((true_vec - mean_duration) ** 2).sum()
    return 1.0 - (numerator / denominator)


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


def test_lr_sick(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    print("总人数", len(label_sick_test))
    print("实际患病人数",np.sum(label_sick_test))
    auc = calculate_auc(label_sick_test, pred_sick_test)
    acc = calculate_accuracy(label_sick_test, pred_sick_test)
    precision = calculate_precision(label_sick_test, pred_sick_test)
    recall = calculate_recall(label_sick_test, pred_sick_test)
    return [auc, acc, precision, recall]


def test_mlp_sick(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
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


def test_lr_jbbm(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    # top1_recall,top2_recall,top3_recall=recall_top(label_jbbm_test, pred_jbbm_test)
    top1_pre, top2_pre, top3_pre = precision_top(label_jbbm_test, pred_jbbm_test)
    # r2_jbbm=calculate_r_squared(label_jbbm_test, pred_jbbm_test)
    return [top1_pre, top2_pre, top3_pre]


def test_mlp_jbbm(filename, month):
    dataset_jbbm_test, dataset_drug_test, label_jbbm_test, label_sick_test = load_test_data(month)
    model = load_model('./data_h5/' + filename)
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    pred_jbbm_test, pred_sick_test = model.predict(x=[dataset_jbbm_test, dataset_drug_test])
    # top1_recall,top2_recall,top3_recall=recall_top(label_jbbm_test, pred_jbbm_test)
    top1_pre, top2_pre, top3_pre = precision_top(label_jbbm_test, pred_jbbm_test)
    # r2_jbbm=calculate_r_squared(label_jbbm_test, pred_jbbm_test)
    return [top1_pre, top2_pre, top3_pre]
