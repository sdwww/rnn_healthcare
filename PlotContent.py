import BaselineTest
import DoctorAITest
import matplotlib.pyplot as plt
import numpy as np


def plot_sick_results():
    model_results = []
    lr_results = []
    mlp_results = []
    # # 测试RNN模型
    month_list = [3, 6, 9, 12]
    for i in month_list:
        model_result = DoctorAITest.test_model_sick('model_20epochs_' + str(i) + 'month.h5', month=3)
        model_results.append(model_result)
    # 测试逻辑回归模型
    for i in month_list:
        lr_result = BaselineTest.test_lr_sick('lr_20epochs_' + str(i) + 'month.h5', month=3)
        lr_results.append(lr_result)

    # 测试多层感知机模型
    for i in month_list:
        mlp_result = BaselineTest.test_mlp_sick('mlp_20epochs_' + str(i) + 'month.h5', month=3)
        mlp_results.append(mlp_result)
    model_results=np.asarray(model_results)
    lr_results=np.asarray(lr_results)
    mlp_results=np.asarray(mlp_results)

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('auc results')
    plt.xlabel('time')
    plt.ylabel('auc')
    plt.plot(month_list, model_results[:,0], 'r', label='rnn')
    plt.plot(month_list, lr_results[:,0], 'b', label='lr')
    plt.plot(month_list, mlp_results[:,0], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.7, 1])
    plt.grid()
    plt.savefig('./data_png/sick_auc.png',dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('accuracy results')
    plt.xlabel('time')
    plt.ylabel('acc')
    plt.plot(month_list, model_results[:,1], 'r', label='rnn')
    plt.plot(month_list, lr_results[:,1], 'b', label='lr')
    plt.plot(month_list, mlp_results[:,1], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.7, 1])
    plt.grid()
    plt.savefig('./data_png/sick_acc.png',dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('precision results')
    plt.xlabel('time')
    plt.ylabel('precision')
    plt.plot(month_list, model_results[:,2], 'r', label='rnn')
    plt.plot(month_list, lr_results[:,2], 'b', label='lr')
    plt.plot(month_list, mlp_results[:,2], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.5, 1])
    plt.grid()
    plt.savefig('./data_png/sick_precision.png',dpi=300)
    plt.close()

    group_labels = ['3months', '6months', '9months', '12months']
    plt.title('recall results')
    plt.xlabel('time')
    plt.ylabel('recall')
    plt.plot(month_list, model_results[:,3], 'r', label='rnn')
    plt.plot(month_list, lr_results[:,3], 'b', label='lr')
    plt.plot(month_list, mlp_results[:,3], 'g', label='mlp')
    plt.xticks(month_list, group_labels, rotation=0)
    plt.legend(bbox_to_anchor=[0.3, 1])
    plt.grid()
    plt.savefig('./data_png/sick_recall.png',dpi=300)
    plt.close()