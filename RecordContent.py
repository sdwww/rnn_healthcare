from openpyxl import load_workbook

import DoctorAITest
import BaselineTest


def record_result():
    # 测试RNN模型
    month_list = [3, 6, 9, 12]
    emb_list = [500, 1000]
    hidden_list = [[300, 300], [300]]
    rnn_uint_list = ['lstm', 'gru']
    wb = load_workbook(filename='./test_result/32test_result.xlsx')
    sheetnames = wb.get_sheet_names()
    ws = wb.get_sheet_by_name(sheetnames[0])
    ws_count = 2
    for month in month_list:
        for emb in emb_list:
            for hidden in hidden_list:
                for rnn_unit in rnn_uint_list:
                    file_name = 'rnn' + str(len(hidden)) + '_' + str(emb) + 'emb_' + str(hidden) \
                                + 'hidden_' + '_' + rnn_unit + '_' + str(20) + 'epochs_' + str(
                        month) + 'month'

                    sick_result = DoctorAITest.test_model_sick(file_name + '.h5', month=month)
                    jbbm_result = DoctorAITest.test_model_jbbm(file_name + '.h5', month=month)
                    ws.cell(column=1, row=ws_count, value=ws_count)
                    ws.cell(column=2, row=ws_count, value='RNN')
                    ws.cell(column=3, row=ws_count, value=emb)
                    ws.cell(column=4, row=ws_count, value=str(hidden))
                    ws.cell(column=5, row=ws_count, value=rnn_unit)
                    ws.cell(column=6, row=ws_count, value=month)
                    ws.cell(column=7, row=ws_count, value=sick_result[0])
                    ws.cell(column=8, row=ws_count, value=sick_result[1])
                    ws.cell(column=9, row=ws_count, value=sick_result[2])
                    ws.cell(column=10, row=ws_count, value=sick_result[3])
                    ws.cell(column=11, row=ws_count, value=sick_result[4])
                    ws.cell(column=12, row=ws_count, value=jbbm_result[0])
                    ws.cell(column=13, row=ws_count, value=jbbm_result[1])
                    ws.cell(column=14, row=ws_count, value=jbbm_result[2])
                    ws_count += 1
                    wb.save('./test_result/32test_result.xlsx')
    for month in month_list:
        sick_result = BaselineTest.test_lr_sick('lr_10epochs_' + str(month) + 'month.h5', month=month)
        jbbm_result = BaselineTest.test_lr_jbbm('lr_10epochs_' + str(month) + 'month.h5', month=month)
        ws.cell(column=1, row=ws_count, value=ws_count)
        ws.cell(column=2, row=ws_count, value='LR')
        ws.cell(column=6, row=ws_count, value=month)
        ws.cell(column=7, row=ws_count, value=sick_result[0])
        ws.cell(column=8, row=ws_count, value=sick_result[1])
        ws.cell(column=9, row=ws_count, value=sick_result[2])
        ws.cell(column=10, row=ws_count, value=sick_result[3])
        ws.cell(column=11, row=ws_count, value=sick_result[4])
        ws.cell(column=12, row=ws_count, value=jbbm_result[0])
        ws.cell(column=13, row=ws_count, value=jbbm_result[1])
        ws.cell(column=14, row=ws_count, value=jbbm_result[2])
        ws_count += 1
        wb.save('./test_result/32test_result.xlsx')
        sick_result = BaselineTest.test_mlp_sick('mlp_10epochs_' + str(month) + 'month.h5', month=month)
        jbbm_result = BaselineTest.test_mlp_jbbm('mlp_10epochs_' + str(month) + 'month.h5', month=month)
        ws.cell(column=1, row=ws_count, value=ws_count)
        ws.cell(column=2, row=ws_count, value='MLP')
        ws.cell(column=6, row=ws_count, value=month)
        ws.cell(column=7, row=ws_count, value=sick_result[0])
        ws.cell(column=8, row=ws_count, value=sick_result[1])
        ws.cell(column=9, row=ws_count, value=sick_result[2])
        ws.cell(column=10, row=ws_count, value=sick_result[3])
        ws.cell(column=11, row=ws_count, value=sick_result[4])
        ws.cell(column=12, row=ws_count, value=jbbm_result[0])
        ws.cell(column=13, row=ws_count, value=jbbm_result[1])
        ws.cell(column=14, row=ws_count, value=jbbm_result[2])
        ws_count += 1
        wb.save('./test_result/32test_result.xlsx')
