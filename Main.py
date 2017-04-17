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
import re

if __name__ == '__main__':
    start = time.clock()
    # # 连接数据库
    # con = DBOptions.connect()
    # cursor = con.cursor()
    # CreateDataset.create_dataset_jbbm(cursor=cursor)
    # cursor.close()
    # con.close()

    # # 训练RNN模型
    # month_list = [3, 6, 9, 12]
    # emb_list = [500, 1000, 1500]
    # hidden_list = [[200, 200], [200]]
    # rnn_uint = ['gru', 'lstm']
    # max_epochs = [10, 20, 30]
    # for i in month_list:
    #     for j in emb_list:
    #         for k in hidden_list:
    #             for l in rnn_uint:
    #                 for m in max_epochs:
    #                     print('./data_h5/rnn' + str(len(k)) + '_emb' + str(j) + '_hidden' + str(k) + '_'+l+'_'
    #                           + str(m) + 'epochs_' + str(i) + 'month.h5')
    #                     # DoctorAITrain.train_model(month=i, max_epochs=20, batch_size=100,
    #                     #                           emb_size=500, hidden_size=j, rnn_unit=k)
    DoctorAITrain.train_model(month=3, max_epochs=2, batch_size=200,emb_size=500,hidden_size=[200])
    # # 测试RNN模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     #DoctorAITest.test_model_sick('model_20epochs_'+str(i)+'month.h5', month=3)
    #     DoctorAITest.test_model_jbbm('rnn2_[200,200]_20epochs_'+str(i)+'month.h5', month=3)
    # print()

    # # 训练逻辑回归模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     BaselineTrain.train_lr(month=i, max_epochs=20, batch_size=64)

    # # 测试逻辑回归模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     #BaselineTest.test_lr_sick('lr_20epochs_'+str(i)+'month.h5', month=3)
    #     BaselineTest.test_lr_jbbm('lr_20epochs_' + str(i) + 'month.h5', month=3)
    # print()
    # # 训练多层感知机模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     BaselineTrain.train_mlp(month=i, max_epochs=20, batch_size=64)

    # #测试多层感知机模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     #BaselineTest.test_mlp_sick('mlp_20epochs_'+str(i)+'month.h5', month=3)
    #     BaselineTest.test_mlp_jbbm('mlp_20epochs_' + str(i) + 'month.h5', month=3)
    # print()

    # # 绘制疾病预测结果
    # PlotContent.plot_sick_results()

    print('总时间:', time.clock() - start)
