import pandas as pd
import time
import datetime
import numpy as np
import DBOptions
import FileOptions
import doctorAI
import re

if __name__ == '__main__':
    start = time.clock()
    # 连接数据库
    con = DBOptions.connect()
    cursor = con.cursor()
    DBOptions.create_index_jbbm_drug(cursor=cursor,con=con)
    cursor.close()
    con.close()
    print()
    print('总时间:', time.clock() - start)
