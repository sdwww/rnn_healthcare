import pandas as pd
import time
import datetime
import numpy as np
import DBOptions
import FileOptions
import CreateDataset
import DoctorAITrain
import DoctorAITest
import BaselineTrain
import BaselineTest
import PlotContent
import RecordContent
import re

if __name__ == '__main__':
    start = time.clock()
    # # 连接数据库
    # con = DBOptions.connect()
    # cursor = con.cursor()
    # CreateDataset.create_dataset_drug_nocost(cursor=cursor)
    # cursor.close()
    # con.close()

    # # 训练RNN模型
    # month_list = [3, 6, 9, 12]
    # emb_list = [500,1000]
    # hidden_list = [[300, 300], [300]]
    # rnn_uint_list = ['simplernn']
    # for month in month_list:
    #     for emb in emb_list:
    #         for hidden in hidden_list:
    #             for rnn_unit in rnn_uint_list:
    #                 DoctorAITrain.train_model(month=month, max_epochs=100,
    #                                           batch_size=100, emb_size=emb,
    #                                           hidden_size=hidden, rnn_unit=rnn_unit)
    # # 测试RNN模型
    # month_list = [3, 6, 9, 12]
    # emb_list = [500, 1000]
    # hidden_list = [[300, 300], [300]]
    # rnn_uint_list = ['gru', 'lstm']
    # for month in month_list:
    #     for emb in emb_list:
    #         for hidden in hidden_list:
    #             for rnn_unit in rnn_uint_list:
    #                 file_name = 'rnn' + str(len(hidden)) + '_' + str(emb) + 'emb_' + str(hidden) \
    #                             + 'hidden_' + '_' + rnn_unit + '_' + str(20) + 'epochs_' + str(
    #                     month) + 'month'
    #                 print(file_name)
    #                 DoctorAITest.test_model_jbbm(file_name + '.h5',month=month)

    # # 训练逻辑回归模型
    # month_list = [3, 6, 9,  12]
    # for i in month_list:
    #     BaselineTrain.train_lr(month=i, max_epochs=10, batch_size=100)
    #
    #
    # # 测试逻辑回归模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     print(BaselineTest.test_lr_sick('lr_10epochs_'+str(i)+'month.h5', month=i))
    #     #BaselineTest.test_lr_jbbm('lr_10epochs_' + str(i) + 'month.h5', month=i)
    # print()

    # #训练多层感知机模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     BaselineTrain.train_mlp(month=i, max_epochs=10, batch_size=64)

    # #测试多层感知机模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     print(BaselineTest.test_mlp_sick('mlp_10epochs_'+str(i)+'month.h5', month=i))
    #     #BaselineTest.test_mlp_jbbm('mlp_10epochs_' + str(i) + 'month.h5', month=i)
    # print()

    # # 绘制疾病预测结果
    # PlotContent.plot_sick_results()

    RecordContent.record_result()
    #PlotContent.plot_jbbm_num()

    print('总时间:', time.clock() - start)
