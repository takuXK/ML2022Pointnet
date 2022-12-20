import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def CloudPointVisualization(pointData, label, title=None, bias="", save_dir="/", savefig=False):
    labelData = pd.DataFrame(
        data={
            "x": pointData[:, 0],
            "y": pointData[:, 1],
            "z": pointData[:, 2],
            "label": label,
        }
    )
    # 单纯绘制点云图像而不管其在物体中所属的部分
    fig1 = plt.figure(figsize=(15, 10))
    ax1 = plt.axes(projection="3d")
    ax1.scatter(labelData["x"], labelData["y"], labelData["z"])
    if title:
        plt.title(title + " with no classification")
    plt.show()
    # 按点云图像各点所属部分分别绘制
    fig2 = plt.figure(figsize=(15, 10))
    ax2 = plt.axes(projection="3d")
    for i in range(label.min(), label.max() + 1):
        c_df = labelData[labelData['label'] == i]
        ax2.scatter(c_df["x"], c_df["y"], c_df["z"])
    # ax.legend()
    if title:
        plt.title(title + " with segmentation")
    if savefig:
        plt.savefig(save_dir + "/" + title + "_" + bias + ".png")
    plt.show()
    
def confusion_matrix_plot(cmatrix_data, seg_num, bias="", save_dir="/", savefig=False):
    normCmatrix = np.zeros(cmatrix_data.shape)
    for i in range(len(cmatrix_data)):
        normCmatrix[i, :] = cmatrix_data[i, :] / cmatrix_data.sum(axis=1)[i]
    cm = pd.DataFrame(data=normCmatrix, index=list(range(seg_num)), columns=list(range(seg_num)))
    fig = plt.figure(figsize=(1*seg_num, 1*seg_num))
    ax = fig.add_subplot(111)
    df_cm = pd.DataFrame(cm)
    sn.heatmap(df_cm, annot=True, vmax=1.0, vmin=0.0, fmt='.3f', cmap='Greys', annot_kws={'size': 10})
    label_x = ax.get_xticklabels()
    label_y = ax.get_yticklabels()
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    if savefig:
        plt.savefig(save_dir + "/" + 'confusion_matrix_' + bias + '.png')
    # if seg_num <= 16:
    #     plt.show()

def plot_accuracy_loss(epochAccuracy, epochLoss, epoches, acc_plot=True, loss_plot=True, bias="",
                       save_dir="/", savefig=True):
    if acc_plot and loss_plot:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ln1 = ax1.plot(np.linspace(1, epoches, epoches), epochAccuracy, label="Accuracy", color='b')
        ax2 = ax1.twinx()
        ln2 = ax2.plot(np.linspace(1, epoches, epoches), epochLoss, label="Loss", color='r')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax2.set_ylabel("Loss")
        lns = ln1 + ln2
        legendtxt = [l.get_label() for l in lns]
        plt.legend(lns, legendtxt, loc=0)
        figname = "Accuracy_Loss_" + bias + ".png"
        if savefig:
            plt.savefig(save_dir + "/" +  figname)
        plt.show()
    elif acc_plot or loss_plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        if acc_plot:
            plot_data = epochAccuracy
            ylabeltxt = "Accuracy"
            linecolor = 'b'
        else:
            plot_data = epochLoss
            ylabeltxt = "Loss"
            linecolor = 'r'
        ax.plot(np.linspace(1, epoches, epoches), plot_data, label=ylabeltxt, color=linecolor)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabeltxt)
        plt.legend()
        figname = ylabeltxt + "_" + bias + ".png"
        if savefig:
            plt.savefigsavefig(save_dir + "/" + figname)
        plt.show()
    else:
        print("Warning: no fig of accuracy and loss ploted.")

def plot_require_data(xdata, ydata, xdatalabel, ydatalebel, bias="", save_dir="/", savefig=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(xdata, ydata, label=ydatalebel, color='black')
    ax.set_xlabel(xdatalabel)
    ax.set_ylabel(ydatalebel)
    plt.legend()
    figname = ydatalebel + "_" + bias + ".png"
    if savefig:
        plt.savefig(save_dir + "/" + figname)
    plt.show()

if __name__ == "__main__":
    # print(os.getcwd())
    filePath = 'F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\data\shapenetcore_partanno_segmentation_benchmark_v0_normal\\04099429\\1ada292665f218a8789f2aa296a418cc.txt'
    # filePath = 'F:\ZJU课程\机器学习（胡浩基）\FinalAssignment\RepeatPointNet\\valpath\dd2d95b294a7a8b4fa5b6212657ae4a4.txt'
    # 读取点云文件
    point_cloud = np.loadtxt(filePath)[:, 0:3]
    label = np.loadtxt(filePath)[:, -1].astype('int')
    label = label - np.min(label)
    print("data load completed.")
    # 绘图
    CloudPointVisualization(point_cloud, label, 'plane')
    print('point cloud shape:{}'.format(point_cloud.shape))
    print('label shape:{}'.format(label.shape))
    print("point construct completed.")