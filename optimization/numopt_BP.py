import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from model import *

os.makedirs("../BPNN", exist_ok=True)
dir_selected = '../BPNN/'


def data_plot(M=None, M_without_noise=None, M_filtered=None, t=None, material=None, ta=None, tb=None, shape=None,
              file=None, filter=None, dir_save=None, label=None):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8, 1e1)
    plt.plot(t, M_without_noise, '--', color="r", label=label + "_noiseless")
    plt.plot(t, M, 'o', color="g", label=label + "_noise")
    plt.plot(t, M_filtered, '*', color="y", label=label + "_filted")
    plt.xlabel("t /s")
    plt.ylabel("Response")
    plt.title("snr=" + "%d" % file + " material=" + material + " ta=" + "%.2f" % ta + " tb=" + "%.2f" % tb +
              " shape=" + "%d" % shape + " filter=" + filter + " " + str(label))
    plt.savefig(dir_save + " " + "snr=" + "%d" % file + " material=" + material + " ta=" + "%.2f" % ta +
                " tb=" + "%.2f" % tb + " shape=" + "%d" % shape + " filter=" + filter + " " + str(label) + '.png')
    plt.savefig(dir_save + " " + "snr=" + "%d" % file + " material=" + material + " ta=" + "%.2f" % ta +
                " tb=" + "%.2f" % tb + " shape=" + "%d" % shape + " filter=" + filter + " " + str(label) + '.eps')
    plt.legend()
    # plt.show()
    plt.close()


# 以信噪比20为例，对数据集进行划分
def data_split():
    data = pd.read_csv('./generate data/noisy signal/M1_response_curve_sum_snr20_M1.csv', )
    # 数据概况预览
    feature_lable = np.array(data)
    feature = feature_lable[:, 3:feature_lable.shape[1]]
    X = feature
    print("开始划分数据集:-------------------------------")
    X_train, X_test = X[:-100], X[-100:]
    pd.DataFrame(X_train).to_csv('./generate data/noisy signal/M1_response_curve_sum_snr20_M1_train.csv')
    pd.DataFrame(X_test).to_csv('./generate data/noisy signal/M1_response_curve_sum_snr20_M1_test.csv')
    return X_train, X_test


# 获取数据
signal_path = './generate data/test1/M1_response_curve_sum_summary_total_train.csv'
signal = np.array(pd.read_csv(signal_path))
signal_noisy = signal[:, 7:207]
signal_clean = signal[:, 212:413]

# 获取测试数据
signal_test_path = './generate data/test1/M1_response_curve_sum_summary_total_test.csv'
signal_test = np.array(pd.read_csv(signal_test_path))
signal_clean_special = signal_test[:1000, 212:413]
signal_noisy_special = signal_test[:1000, 7:207]
ta = signal_test[:1000, 2:3]
tb = signal_test[:1000, 3:4]
material_num = signal_test[:1000, 4:5]
snr = signal_test[:1000, 5:6]
shape_flag = signal_test[:1000, 6:7]

# --------------------------------------------------------------------------------------------
model = BP() # 使用不同优化算法的神经网络
model.compile(loss='mean_squared_error', optimizer='Adam')  # 编译模型
model.fit(signal_noisy, signal_clean, epochs=500, batch_size=128)  # 训练模型nb_epoch=50次
model.summary()  # 模型描述

# 在训练集上的拟合结果
y_train_predict = model.predict(signal_noisy)
y_test_predict = model.predict(signal_noisy_special)
pd.DataFrame(y_test_predict).to_csv('./generate data/test1/' + 'y_test_predict.csv')

pd.DataFrame(np.vstack((signal_clean_special, signal_noisy_special, y_test_predict))).to_csv('./generate data/test1/' + 'signal_test.csv')

def BP_run():
    t = np.array(10 ** (np.linspace(-8, 0, 200)))
    material = ''
    lable = 'M1'
    success_list = []

    for i in range(len(y_test_predict)):
        print(i)

        filted_result = ''
        lable = 'M1'
        M = signal_noisy_special[i]
        M_without_noise = signal_clean_special[i]
        M_filtered = y_test_predict[i]
        ta_special = ta[i]
        material_num_special = material_num[i]
        tb_special = tb[i]
        snr_special = snr[i]
        shape_flag_special = shape_flag[i]

        if material_num_special == 0:
            material = 'Steel'
        if material_num_special == 1:
            material = 'Ni'
        if material_num_special == 2:
            material = 'Al'


        data_plot(M=M, M_without_noise=M_without_noise, M_filtered=M_filtered, t=t, material=material, ta=ta_special, tb=tb_special,
                  shape=shape_flag_special, file=snr_special, filter='BPNN-OP', dir_save=dir_selected, label=lable)

        # 保存数据
        snr_denoised = 10 * np.log10(np.sum(M ** 2) / np.sum((M - np.array(M_filtered)) ** 2))
        M_snr = np.hstack((snr_special, M))
        M_without_noise_snr = np.hstack(('None', M_without_noise))
        M_filtered_snr = np.hstack((snr_denoised, M_filtered))
        if snr_denoised >= snr_special:
            success = "snr=" + "%d" % snr_special + " material=" + material + " ta=" + "%.2f" % ta_special + " tb=" + "%.2f" % tb_special + " shape=" + "%d" % shape_flag_special + " filter=BPNN-OP " + lable
            success = np.hstack(
                (snr_special, snr_denoised, material, ta_special, tb_special, shape_flag_special, 'BPNN', lable))
            if len(success_list) == 0:
                success_list = success
            else:
                success_list = np.vstack((success_list, success))
        filted_result = np.vstack((M_without_noise, M, M_filtered))
        pd.DataFrame(filted_result, index=['signal_clean', 'signal_noisy', 'filted']).to_csv(dir_selected + " " + "snr=" + "%d" % snr_special + " material=" + material + " ta=" +
                                           "%.2f" % ta_special + " tb=" + "%.2f" % tb_special + " shape=" + "%d" % shape_flag_special +
                                           " filter=BPNN-OP " + lable + '.csv')
    return

if __name__ == '__main__':
    BP_run()
    # if i >= 2:
    #     break

