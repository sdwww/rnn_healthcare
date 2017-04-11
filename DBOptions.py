import os
import datetime

from openpyxl import load_workbook

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
import cx_Oracle as db
import numpy as np
import FileOptions
np.random.seed(1234)

# 连接数据库
def connect():
    con = db.connect('MH3', '123456', '127.0.0.1:1521/ORCL')
    return con


# 执行select语句
def getSQL(sql, cursor):
    cursor.execute(sql)
    result = cursor.fetchall()
    content = []
    for row in result:
        if len(row) == 1:
            content.append(row[0])
        else:
            content.append(list(row))
    return content


# 执行更新操作
def exeSQL(sql, cursor, con):
    cursor.execute(sql)
    con.commit()
    return 1


# 获取数据库的部分内容
def getAllInfo(cursor, spm_list, jbmc_list):
    all_info = []
    for i in range(49):
        sql = 'SELECT * from MH3.DATA_ANALYSIS where rownum <=' + str(1000000 + i * 1000000) + \
              ' MINUS SELECT * from DATA_ANALYSIS where rownum <= ' + str(i * 1000000)
        batch_info = getSQL(sql, cursor)
        for i in batch_info:
            if i[4] in jbmc_list and i[5] in spm_list:
                all_info.append(i)
    return all_info


# 将与药典中商品名匹配的记录插入新表中
def match_spm_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/SPM_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    print(ws.max_row)
    count = 0
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w2 = ws.cell(row=rx, column=2).value
        w3 = ws.cell(row=rx, column=3).value
        w4 = ws.cell(row=rx, column=4).value
        if w4:
            sql = "insert into DATA_ANALYSIS_FINAL1 select * from DATA_ANALYSIS where spm='" + w1 + "'"
            exeSQL(sql, cursor, con)
            count += 1
            print(count)


# 根据JBBM_JBBMC_XOUNT.xlsx的内容筛选出匹配的疾病编码和疾病名称
def match_jbbm_jbmc_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/SPM_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    print(ws.max_row)
    count = 0
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w2 = ws.cell(row=rx, column=2).value
        w4 = ws.cell(row=rx, column=4).value
        if w4:
            sql = "insert into DATA_ANALYSIS_FINAL2 select * from DATA_ANALYSIS_FINAL1 " \
                  "where JBBM='" + w1 + "' AND JBMC='" + w2 + "'"
            exeSQL(sql, cursor, con)
            count += 1
            print(count)


# 将与药典中商品名匹配的药品名更新到新表中
def update_drug_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/SPM_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    print(ws.max_row)
    count = 0
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w2 = ws.cell(row=rx, column=2).value
        w3 = ws.cell(row=rx, column=3).value
        w4 = ws.cell(row=rx, column=4).value
        if w4:
            sql = "update DATA_ANALYSIS_FINAL3 set DRUG='" + w4 + "' where spm='" + w1 + "'"
            exeSQL(sql, cursor, con)
            count += 1
            print(count)


# 根据JBBM_JBBMC_XOUNT.xlsx的内容更新匹配的疾病编码和疾病名称
def update_jbbm_jbmc_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/JBBM_JBMC_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    print(ws.max_row)
    count = 0
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w2 = ws.cell(row=rx, column=2).value
        w4 = ws.cell(row=rx, column=4).value
        w5 = ws.cell(row=rx, column=5).value
        if w4:
            sql = "UPDATE DATA_ANALYSIS_FINAL3 SET JBBM='" + w4 \
                  + "',JBMC=" + "'" + w5 + "' where JBBM_OLD='" + str(
                w1) + "' AND JBMC_OLD='" + w2 + "'"
            # exeSQL(sql, cursor, con)
            print(sql)
            count += 1
            print(count)


# 在数据库添加疾病类别
def update_jbmccateg_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/JBBM_JBMC_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    delete_count = 0
    drug_list = []
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w4 = ws.cell(row=rx, column=4).value
        if w4 and w4 not in drug_list:
            drug_list.append(w4)
    drug_categ_dict = {}
    wb = load_workbook(filename='./data_xlsx/3位代码类目表（ICD-10）.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    for rx in range(2, ws.max_row + 1):
        w1 = ws.cell(row=rx, column=1).value
        w2 = ws.cell(row=rx, column=2).value
        for i in drug_list:
            if w1[0:3] == i[0:3]:
                drug_categ_dict[i] = w2
    count = 0
    for i, j in drug_categ_dict.items():
        sql = "UPDATE DATA_ANALYSIS_FINAL3 SET JBMC_CATEG='" + j + "' where JBBM='" + i + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)


# 将与药典中药品名匹配的药品名分类更新到新表中
def update_drugcateg_db(cursor, con):
    wb = load_workbook(filename='./data_xlsx/SPM_COUNT.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    print(ws.max_row)
    count = 0
    drug_categ_dict = {}
    for rx in range(2, ws.max_row + 1):
        w4 = ws.cell(row=rx, column=4).value
        w5 = ws.cell(row=rx, column=5).value
        if w4 and w4 not in drug_categ_dict:
            drug_categ_dict[w4] = w5
    for i, j in drug_categ_dict.items():
        sql = "update DATA_ANALYSIS_FINAL3 set DRUG_CATEG='" + j + "' where DRUG='" + i + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)


# 对数据库添加就医次数和距上一次时间间隔索引
def create_index_duration(cursor, con):
    sql = "SELECT XH,GRBH,ZYRQ FROM DATA_ANALYSIS_JBBM WHERE " \
          "ZYRQ <= to_date('2014-07-29 00:00:00','yyyy-mm-dd hh24:mi:ss') ORDER BY ZYRQ DESC"
    dataset = getSQL(sql, cursor)
    pidAdmMap = {}
    admDateMap = {}
    for i in dataset:
        admId = int(i[0])
        pid = str(i[1])
        admTime = i[2]
        admDateMap[admId] = admTime  # 住院id和住院时间
        if pid in pidAdmMap:
            pidAdmMap[pid].append([admId, admTime])
        else:
            pidAdmMap[pid] = [[admId, admTime]]
    count = 0
    admIndex = {}
    admPeriod = {}
    for i in pidAdmMap:
        for j in pidAdmMap[i]:
            index = pidAdmMap[i].index(j)
            admIndex[j[0]] = index
            if pidAdmMap[i].index(j) == len(pidAdmMap[i]) - 1:
                admPeriod[j[0]] = 0
            else:
                admPeriod[j[0]] = (pidAdmMap[i][index][1] - pidAdmMap[i][index + 1][1]).days
    # for i in admPeriod:
    #     sql = "update MH3.DATA_ANALYSIS_JBBM set DURATION=" + str(admPeriod[i]) + " where XH=" + str(i)
    #     result=exeSQL(sql, cursor, con)
    #     if result:
    #         count += 1
    #         print(count)
    for i in admIndex:
        sql = "update MH3.DATA_ANALYSIS_JBBM set XH_INDEX=" + str(admIndex[i] + 1) + " where XH=" + str(i)
        result = exeSQL(sql, cursor, con)
        if result:
            count += 1
            if count % 100 == 0:
                print(count)


# 对数据库个人编码添加索引
def create_index_grbh(cursor, con):
    sql = 'SELECT GRBH FROM DATA_ANALYSIS_JBBM group by GRBH'
    grbh_all = getSQL(sql, cursor)
    random_rank=np.argsort(np.random.randint(1, 1000000, len(grbh_all)))
    grbhIndex = {}
    for i in range(len(grbh_all)):
        grbhIndex[grbh_all[i]] = random_rank[i]
    count = 0
    for i in grbhIndex:
        sql = "update MH3.DATA_ANALYSIS_JBBM set GRBH_INDEX=" + str(grbhIndex[i]) + " where GRBH='" + str(i) + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)


# 对数据库添加疾病编码和药品名的索引
def create_index_jbbm_drug(cursor, con):
    sql = 'SELECT JBBM FROM DATA_ANALYSIS_JBBM group by JBBM'
    jbbm_all = getSQL(sql, cursor)
    sql = 'SELECT DRUG FROM DATA_ANALYSIS_DRUG group by DRUG'
    drug_all = getSQL(sql, cursor)
    jbbmIndex = {}
    for i in range(len(jbbm_all)):
        jbbmIndex[jbbm_all[i]] = i
    count = 0
    for i in jbbmIndex:
        sql = "update MH3.DATA_ANALYSIS_JBBM set JBBM_INDEX=" + str(jbbmIndex[i]) + " where JBBM='" + str(i) + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)
    drugIndex = {}
    for i in range(len(drug_all)):
        drugIndex[drug_all[i]] = i
    for i in drugIndex:
        sql = "update MH3.DATA_ANALYSIS_DRUG set DRUG_INDEX=" + str(drugIndex[i]) + " where DRUG='" + str(i) + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)


# 对数据库添加分类的索引
def create_index_categ(cursor, con):
    sql = 'SELECT JBMC_CATEG FROM DATA_ANALYSIS_JBBM group by JBMC_CATEG'
    jbmc_categ_all = getSQL(sql, cursor)
    sql = 'SELECT DRUG_CATEG FROM DATA_ANALYSIS_DRUG group by DRUG_CATEG'
    drug_categ_all = getSQL(sql, cursor)
    count = 0
    jbmccategIndex = {}
    for i in range(len(jbmc_categ_all)):
        jbmccategIndex[jbmc_categ_all[i]] = i
    for i in jbmccategIndex:
        sql = "update MH3.DATA_ANALYSIS_JBBM set JBMC_CATEG_INDEX=" + str(
            jbmccategIndex[i]) + " where JBMC_CATEG='" + str(i) + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)

    drugcategIndex = {}
    for i in range(len(drug_categ_all)):
        drugcategIndex[drug_categ_all[i]] = i
    for i in drugcategIndex:
        sql = "update MH3.DATA_ANALYSIS_DRUG set DRUG_CATEG_INDEX=" + str(
            drugcategIndex[i]) + " where DRUG_CATEG='" + str(
            i) + "'"
        exeSQL(sql, cursor, con)
        count += 1
        print(count)
