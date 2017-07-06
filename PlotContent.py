import BaselineTest
import DoctorAITest
import matplotlib.pyplot as plt
import numpy as np
import DBOptions


def plot_sick_results():
    model_results = []
    lr_results = []
    mlp_results = []
    # # 测试RNN模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        model_result = DoctorAITest.test_model_sick('rnn1_500emb_[300]hidden__gru_20epochs_' + str(i) + 'month.h5',
                                                    month=3)
        model_results.append(model_result)
    # 测试逻辑回归模型
    for i in month_list:
        lr_result = BaselineTest.test_lr_sick('lr_10epochs_' + str(i) + 'month.h5', month=3)
        lr_results.append(lr_result)

    # 测试多层感知机模型
    for i in month_list:
        mlp_result = BaselineTest.test_mlp_sick('mlp_10epochs_' + str(i) + 'month.h5', month=3)
        mlp_results.append(mlp_result)
    model_results = np.asarray(model_results)
    lr_results = np.asarray(lr_results)
    mlp_results = np.asarray(mlp_results)

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('auc results')
    plt.xlabel('time')
    plt.ylabel('auc')
    plt.plot(month_list, model_results[:, 0], 'r', label='rnn')
    plt.plot(month_list, lr_results[:, 0], 'b', label='lr')
    plt.plot(month_list, mlp_results[:, 0], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.7, 1])
    plt.grid()
    plt.savefig('./data_png/sick_auc.png', dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('accuracy results')
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.plot(month_list, model_results[:, 1], 'r', label='rnn')
    plt.plot(month_list, lr_results[:, 1], 'b', label='lr')
    plt.plot(month_list, mlp_results[:, 1], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.7, 1])
    plt.grid()
    plt.savefig('./data_png/sick_acc.png', dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('precision results')
    plt.xlabel('time')
    plt.ylabel('precision')
    plt.plot(month_list, model_results[:, 2], 'r', label='rnn')
    plt.plot(month_list, lr_results[:, 2], 'b', label='lr')
    plt.plot(month_list, mlp_results[:, 2], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.5, 1])
    plt.grid()
    plt.savefig('./data_png/sick_precision.png', dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('recall results')
    plt.xlabel('time')
    plt.ylabel('recall')
    plt.plot(month_list, model_results[:, 3], 'r', label='rnn')
    plt.plot(month_list, lr_results[:, 3], 'b', label='lr')
    plt.plot(month_list, mlp_results[:, 3], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.savefig('./data_png/sick_recall.png', dpi=300)
    plt.close()


def plot_jbbm_num():
    month_list = [3, 6, 9, 12]
    for month in month_list:
        label_jbbm_train = np.load('./data_npz/label_jbbm_train_' + str(month) + 'month.npz')["arr_0"]
        label_jbbm_num = np.sum(label_jbbm_train, axis=1)
        max_num = np.max(label_jbbm_num)
        min_num = np.min(label_jbbm_num)
        jbbm_count = np.zeros([max_num - min_num + 1])
        for i in label_jbbm_num:
            jbbm_count[i] += 1
        print(jbbm_count)
        plt.bar(range(len(jbbm_count)), jbbm_count)
        plt.savefig('./data_png/label_jbbm_num' + str(month) + 'month.png', dpi=300)
        plt.close()


def plot_number_xh(cursor):
    sql = 'select count(*) as a from DATA_ANALYSIS_JBBM where XH_INDEX>0 group by grbh order by count(*)'
    result = DBOptions.getSQL(sql, cursor)
    max_num = np.max(result)
    min_num = np.min(result)
    xh_count = np.zeros([max_num - min_num + 2])
    for i in result:
        xh_count[i] += 1
    print(xh_count)
    plt.bar(range(len(xh_count)), xh_count)
    plt.savefig('./data_png/number_xh.png', dpi=300)
    plt.close()
