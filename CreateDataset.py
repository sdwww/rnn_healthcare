import numpy as np
import DBOptions

patient_num = 28753
visit_num = 40
jbbm_num = 1236
drug_num = 1096
jbbm_categ_num = 588


# 创建诊断数据集
def create_dataset_jbbm(cursor):
    sql = 'select grbh_index,xh_index,jbbm_index from DATA_ANALYSIS_JBBM where XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_jbbm = np.zeros((patient_num, visit_num, jbbm_num), dtype='int32')
    print('诊断数据集大小：', dataset_jbbm.shape)
    for i in all_info:
        if i[1] < visit_num:
            dataset_jbbm[i[0], -i[1], i[2]] = 1
    dataset_jbbm_train = dataset_jbbm[:int(patient_num * 0.85)]
    dataset_jbbm_test = dataset_jbbm[int(patient_num * 0.85):]
    np.savez_compressed('./data_npz/dataset_jbbm_train.npz', dataset_jbbm_train)
    np.savez_compressed('./data_npz/dataset_jbbm_test.npz', dataset_jbbm_test)

    time_list = ['2014-10-29', '2015-01-29', '2015-04-29', '2015-07-29']
    for i in range(len(time_list)):
        sql = "select grbh_index,jbmc_categ_index from DATA_ANALYSIS_JBBM where XH_INDEX=0 " \
              "and ZYRQ <= to_date('" + time_list[i] + " 00:00:00','yyyy-mm-dd hh24:mi:ss')"
        all_info = DBOptions.get_sql(sql, cursor=cursor)
        label_jbbm = np.zeros((patient_num, jbbm_categ_num), dtype='int16')
        label_sick = np.zeros((patient_num, 1), dtype='int16')
        print('诊断标签大小：', label_jbbm.shape)
        for item in all_info:
            label_jbbm[item[0], item[1]] = 1
            label_sick[item[0], 0] = 1
        label_jbbm_train = label_jbbm[:int(patient_num * 0.85)]
        label_jbbm_test = label_jbbm[int(patient_num * 0.85):]
        np.savez_compressed('./data_npz/label_jbbm_train_' + str(12 // len(time_list) * (i + 1))
                            + 'month.npz', label_jbbm_train)
        np.savez_compressed('./data_npz/label_jbbm_test_' + str(12 // len(time_list) * (i + 1))
                            + 'month.npz', label_jbbm_test)

        label_sick_train = label_sick[:int(patient_num * 0.85)]
        label_sick_test = label_sick[int(patient_num * 0.85):]
        np.savez_compressed('./data_npz/label_sick_train_' + str(12 // len(time_list) * (i + 1))
                            + 'month.npz', label_sick_train)
        np.savez_compressed('./data_npz/label_sick_test_' + str(12 // len(time_list) * (i + 1))
                            + 'month.npz', label_sick_test)


# 创建药品数据集，内容为药品的花费
def create_dataset_drug(cursor):
    sql = 'select grbh_index,xh_index,DRUG_INDEX,ZJEITEM from DATA_ANALYSIS_DRUG,DATA_ANALYSIS_JBBM where ' \
          'DATA_ANALYSIS_JBBM.XH =DATA_ANALYSIS_DRUG.XH AND DATA_ANALYSIS_JBBM.XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_drug_nocost = np.zeros((patient_num, visit_num, drug_num), dtype='int16')
    print('诊断数据集大小：', dataset_drug_nocost.shape)
    for i in all_info:
        if i[1] < visit_num:
            dataset_drug_nocost[i[0], -i[1], i[2]] += i[3]
    dataset_drug_nocost_train = dataset_drug_nocost[:int(patient_num * 0.85)]
    dataset_drug_nocost_test = dataset_drug_nocost[int(patient_num * 0.85):]
    np.savez_compressed('./data_npz/dataset_drug_nocost_train.npz', dataset_drug_nocost_train)
    np.savez_compressed('./data_npz/dataset_drug_nocost_test.npz', dataset_drug_nocost_test)

    # time_list = ['2014-10-29', '2015-01-29', '2015-04-29', '2015-07-29']
    # for i in range(len(time_list)):
    #     sql = "select grbh_index,DRUG_CATEG_INDEX,ZJEITEM from DATA_ANALYSIS_DRUG,DATA_ANALYSIS_JBBM where " \
    #           "DATA_ANALYSIS_JBBM.XH =DATA_ANALYSIS_DRUG.XH AND DATA_ANALYSIS_JBBM.XH_INDEX=0 " \
    #           "and DATA_ANALYSIS_DRUG.ZYRQ <= to_date('" + time_list[i] + " 00:00:00','yyyy-mm-dd hh24:mi:ss') "
    #     all_info = DBOptions.getSQL(sql, cursor=cursor)
    #     label_drug_nocost = np.zeros((patient_num, drug_categ_num), dtype='int16')
    #     print('药品标签大小：', label_drug_nocost.shape)
    #     for item in all_info:
    #         label_drug_nocost[item[0], item[1]] += item[2]
    #     label_drug_nocost_train = label_drug_nocost[:int(patient_num * 0.85)]
    #     label_drug_nocost_test = label_drug_nocost[int(patient_num * 0.85):]
    #     np.savez_compressed('./data_npz/label_drug_nocost_train_' + str(12 // len(time_list) * (i + 1))
    #                         + 'month.npz', label_drug_nocost_train)
    #     np.savez_compressed('./data_npz/label_drug_nocost_test_' + str(12 // len(time_list) * (i + 1))
    #                         + 'month.npz', label_drug_nocost_test)


# 创建药品数据集，内容为药品是否使用，标记为0-1
def create_dataset_drug_nocost(cursor):
    sql = 'select grbh_index,xh_index,DRUG_INDEX from DATA_ANALYSIS_DRUG,DATA_ANALYSIS_JBBM where ' \
          'DATA_ANALYSIS_JBBM.XH =DATA_ANALYSIS_DRUG.XH AND DATA_ANALYSIS_JBBM.XH_INDEX!=0'
    all_info = DBOptions.get_sql(sql, cursor=cursor)
    dataset_drug_nocost = np.zeros((patient_num, visit_num, drug_num), dtype='int16')
    print('诊断数据集大小：', dataset_drug_nocost.shape)
    for i in all_info:
        if i[1] < visit_num:
            dataset_drug_nocost[i[0], -i[1], i[2]] = 1
    dataset_drug_nocost_train = dataset_drug_nocost[:int(patient_num * 0.85)]
    dataset_drug_nocost_test = dataset_drug_nocost[int(patient_num * 0.85):]
    np.savez_compressed('./data_npz/dataset_drug_nocost_train.npz', dataset_drug_nocost_train)
    np.savez_compressed('./data_npz/dataset_drug_nocost_test.npz', dataset_drug_nocost_test)

    # time_list = ['2014-10-29', '2015-01-29', '2015-04-29', '2015-07-29']
    # for i in range(len(time_list)):
    #     sql = "select grbh_index,DRUG_CATEG_INDEX from DATA_ANALYSIS_DRUG,DATA_ANALYSIS_JBBM where " \
    #           "DATA_ANALYSIS_JBBM.XH =DATA_ANALYSIS_DRUG.XH AND DATA_ANALYSIS_JBBM.XH_INDEX=0 " \
    #           "and DATA_ANALYSIS_DRUG.ZYRQ <= to_date('" + time_list[i] + " 00:00:00','yyyy-mm-dd hh24:mi:ss') "
    #     all_info = DBOptions.getSQL(sql, cursor=cursor)
    #     label_drug_nocost = np.zeros((patient_num, drug_categ_num), dtype='int16')
    #     print('药品标签大小：', label_drug_nocost.shape)
    #     for item in all_info:
    #         label_drug_nocost[item[0], item[1]] = 1
    #     label_drug_nocost_train = label_drug_nocost[:int(patient_num * 0.85)]
    #     label_drug_nocost_test = label_drug_nocost[int(patient_num * 0.85):]
    #     np.savez_compressed('./data_npz/label_drug_nocost_train_' + str(12 // len(time_list) * (i + 1))
    #                         + 'month.npz', label_drug_nocost_train)
    #     np.savez_compressed('./data_npz/label_drug_nocost_test_' + str(12 // len(time_list) * (i + 1))
    #                         + 'month.npz', label_drug_nocost_test)
