from keras.layers import Input, Dense, concatenate, GRU, Dropout
from keras.models import Model

from keras.utils.vis_utils import plot_model
import numpy as np
import xgboost as xgb

import CreateDataset


def change_to_one_zero(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if dataset[i][j] != 0:
                dataset[i][j] = 1
    return dataset


def load_train_data(month):
    dataset_jbbm_train = change_to_one_zero(np.sum(np.load('./data_npz/dataset_jbbm_train.npz')["arr_0"], axis=1))
    dataset_drug_train = change_to_one_zero(
        np.sum(np.load('./data_npz/dataset_drug_nocost_train.npz')["arr_0"], axis=1))
    label_jbbm_train = np.load('./data_npz/label_jbbm_train_' + str(month) + 'month.npz')["arr_0"]
    label_drug_train = np.load('./data_npz/label_drug_nocost_train_' + str(month) + 'month.npz')["arr_0"]
    label_sick_train = np.load('./data_npz/label_sick_train_' + str(month) + 'month.npz')["arr_0"]
    return dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train


def train_lr(month=3, max_epochs=10, batch_size=100):
    jbbm_input = Input(shape=(CreateDataset.jbbm_num,), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.drug_num,), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    sick_output = Dense(1, activation='sigmoid', name='sick_output')(code_input)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'sick_output': 'binary_crossentropy'},
                  loss_weights={'sick_output': 1})

    # 模型可视化
    plot_model(model, to_file='./data_png/lr_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train = load_train_data(
        month)
    model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_sick_train],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/lr_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')


def train_mlp(month=3, max_epochs=10, batch_size=100):
    jbbm_input = Input(shape=(CreateDataset.jbbm_num,), name='jbbm_input')
    drug_input = Input(shape=(CreateDataset.drug_num,), name='drug_input')
    code_input = concatenate([jbbm_input, drug_input])
    hidden_layer = Dense(500)(code_input)
    sick_output = Dense(1, activation='sigmoid', name='sick_output')(hidden_layer)
    model = Model(inputs=[jbbm_input, drug_input], outputs=[sick_output])
    model.compile(optimizer='rmsprop',
                  loss={'sick_output': 'binary_crossentropy'},
                  loss_weights={'sick_output': 1})

    # 模型可视化
    plot_model(model, to_file='./data_png/mlp_model.png', show_shapes=True)

    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train = load_train_data(
        month)
    model.fit(x=[dataset_jbbm_train, dataset_drug_train], y=[label_sick_train],
              epochs=max_epochs, batch_size=batch_size, validation_split=0.2)
    model.save('./data_h5/mlp_' + str(max_epochs) + 'epochs_' + str(month) + 'month.h5')


def train_xgb(month,max_epochs=10):
    dataset_jbbm_train, dataset_drug_train, label_jbbm_train, label_drug_train, label_sick_train \
        = load_train_data(month)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 多分类的问题
        'num_class': 2,
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        'nthread': 7,  # cpu 线程数
        'eval_metric': 'auc'
    }

    plst = list(params.items())
    xgb_train = xgb.DMatrix(np.hstack((dataset_jbbm_train, dataset_drug_train)),
                      label_sick_train)
    model = xgb.train(params=plst,dtrain=xgb_train, num_boost_round=max_epochs)
    model.save_model('xgb.model')  # 用于存储训练出的模型
    return
