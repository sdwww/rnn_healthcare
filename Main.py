import pandas as pd
import time
import datetime
import numpy as np
import DBOptions
import FileOptions
import CreateDataset
import DoctorAITrain
import re

if __name__ == '__main__':
    start = time.clock()
    # # 连接数据库
    # con = DBOptions.connect()
    # cursor = con.cursor()
    # CreateDataset.create_dataset_jbbm(cursor=cursor)
    # cursor.close()
    # con.close()
    DoctorAITrain.doctorAI_train()
    print('总时间:', time.clock() - start)
