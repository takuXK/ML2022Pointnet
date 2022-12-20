import os
# select CUDA core before run the train code if there are multiple GPU accessible
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm
from pointnet_dataset import get_dataLoader
from related_func import *
from pointnet_model import Model_PointNet
from data_visualization import CloudPointVisualization

main_path = "..\\data\\hdf5_data\\"
visualpath = main_path + "test.txt"

modelpath = "..\\Pointnet2022ML\\log\\checkpoints\\"
modelname = "best_model.pth"

savepath = "..\\Pointnet2022ML\\log\\reconstruct_fig"

pointNUM = 2048
batch_size = 2
num_classes = 16
num_part = 50

test_loader = get_dataLoader(None, visualpath, train=False, batch_size=batch_size)

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


'''MODEL LOADING'''
net = Model_PointNet(num_part, normal_channel=False).cuda()
checkpoint = torch.load(modelpath + modelname)
net.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    test_metrics = {}
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    net.eval()

    cur_pred_val_total = []
    cloud_total = []
    pid_total = []
    for cloud, label, pid in tqdm.tqdm(test_loader):
        cur_batch_size, pointnum, _ = cloud.size()

        cloud = cloud.data.numpy()
        cloud[:, :, 0:3] = random_scale_point_cloud(cloud[:, :, 0:3])
        cloud[:, :, 0:3] = shift_point_cloud(cloud[:, :, 0:3])
        cloud = torch.Tensor(cloud)
        cloud, label, pid = cloud.float().cuda(), label.reshape((label.size(0), 1)).long().cuda(), pid.long().cuda()
        cloud = cloud.transpose(2, 1)
        seg_pred, _ = net(cloud, to_categorical(label, num_classes))
        cur_pred_val = seg_pred.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, pointnum)).astype(np.int32)
        pid = pid.cpu().data.numpy()

        for i in range(cur_batch_size):
            cat = seg_label_to_cat[pid[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

        cur_pred_val_total.append(flatten_first_dim(cur_pred_val))
        cloud_total.append(flatten_first_dim(np.array(torch.Tensor.cpu(cloud))))
        pid_total.append(flatten_first_dim(pid))

        correct = np.sum(cur_pred_val == pid)
        total_correct += correct
        total_seen += (cur_batch_size * pointnum)

        # 统计各part数据：输入的数量，识别正确的数量
        for l in range(num_part):
            total_seen_class[l] += np.sum(pid == l)
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (pid == l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = pid[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))  # 统计各batch下不同class的iou

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])  # 以batch做平均，得到本次epoch下不同class的mIOU
    mean_shape_ious = np.mean(list(shape_ious.values()))  # 本次epoch下所有class的IOU之和/class数量
    test_metrics['accuracy'] = total_correct / float(total_seen)
    test_metrics['class_avg_accuracy'] = np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
    # for cat in sorted(shape_ious.keys()):
    #     print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
    test_metrics['class_avg_iou'] = mean_shape_ious
    test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)  # 本次epoch下的mIOU

    # flatten all data
    cur_pred_val_total = np.transpose(flatten_first_dim(cur_pred_val_total))
    cloud_total = np.transpose(flatten_first_dim(cloud_total))
    pid_total = np.transpose(flatten_first_dim(pid_total))

    data_total = np.hstack((cloud_total, cur_pred_val_total.reshape((cur_pred_val_total.size, 1)), pid_total.reshape((pid_total.size, 1))))

    # statistic point
    seg_pointlist = {}
    for j in range(num_part):
        index = np.where(data_total[:, -1] == j)[0]
        seg_pointlist[str(j)] = data_total[index, :]

    for classname in seg_classes.keys():
        class_data = []
        for n in seg_classes[classname]:
            class_data.append(np.transpose(seg_pointlist[str(n)]))
        class_data = np.transpose(flatten_first_dim(class_data))

        CloudPointVisualization(pointData=class_data[:, 0:3], label=class_data[:,-2].astype(np.int32),
                                title=classname, bias="final", savefig=True, save_dir=str(savepath))