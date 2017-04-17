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
    # for i in month_list:
    #     DoctorAITrain.train_model(month=i, max_epochs=20, batch_size=64)

    # 测试RNN模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        #DoctorAITest.test_model_sick('model_20epochs_'+str(i)+'month.h5', month=3)
        DoctorAITest.test_model_jbbm('model_20epochs_'+str(i)+'month.h5', month=3)
    print()
    # # 训练逻辑回归模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     BaselineTrain.train_lr(month=i, max_epochs=20, batch_size=64)

    # 测试逻辑回归模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        #BaselineTest.test_lr_sick('lr_20epochs_'+str(i)+'month.h5', month=3)
        BaselineTest.test_lr_jbbm('lr_20epochs_' + str(i) + 'month.h5', month=3)
    print()
    # # 训练多层感知机模型
    # month_list = [3, 6, 9, 12]
    # for i in month_list:
    #     BaselineTrain.train_mlp(month=i, max_epochs=20, batch_size=64)

    #测试多层感知机模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        #BaselineTest.test_mlp_sick('mlp_20epochs_'+str(i)+'month.h5', month=3)
        BaselineTest.test_mlp_jbbm('mlp_20epochs_' + str(i) + 'month.h5', month=3)
    print()

    # # 绘制疾病预测结果
    # PlotContent.plot_sick_results()

    print('总时间:', time.clock() - start)