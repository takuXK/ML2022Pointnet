import os
# select CUDA core before run the train code if there are multiple GPU accessible
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from sklearn.metrics import confusion_matrix
from pathlib import Path
import datetime
import numpy as np
import tqdm
from pointnet_model import Model_PointNet, LossFunction
from pointnet_dataset import get_dataLoader
from related_func import *
from data_visualization import *



main_path = "..\\data\\hdf5_data\\"
train_txt_path = main_path + "train_hdf5_file_list.txt"
valid_txt_path = main_path + "val_hdf5_file_list.txt"

exp_dir = Path('./log/')
exp_dir.mkdir(exist_ok=True)
checkpoints_dir = exp_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
outfig_dir = exp_dir.joinpath('result_fig/')
outfig_dir.mkdir(exist_ok=True)

epoches = int(input("Please input epoches: "))
pointNUM = 2048
batch_size = int(input("Please input batch size: "))
num_classes = 16
num_part = 50
learning_rate = 0.001

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

train_loader = get_dataLoader(train_txt_path, valid_txt_path, train=True, batch_size=batch_size)
test_loader = get_dataLoader(train_txt_path, valid_txt_path, train=False, batch_size=batch_size)

net = Model_PointNet(normal_channel=False).cuda()
net.apply(inplace_relu)
net.apply(weights_init)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
# loss_function = nn.CrossEntropyLoss()
loss_function = LossFunction().cuda()

best_acc = 0
global_epoch = 0
best_class_avg_iou = 0
best_instance_avg_iou = 0

train_result = {
    "train_accuracy": np.zeros((epoches, 1)),
    "accuracy": np.zeros((epoches, 1)),
    "class mIOU": [],
    "mIOU": np.zeros((epoches, 1)),
    "mIOUc": np.zeros((epoches, 1)),
    "loss": np.zeros((epoches, 1))
}

for epoch in range(epoches):
    print("Running Epoch: " + str(epoch + 1))
    mean_correct = []
    time_start = datetime.datetime.now()
    net.train()
    # train process
    total_loss_epoch = 0
    for cloud, label, pid in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        cloud = cloud.data.numpy()
        cloud[:, :, 0:3] = random_scale_point_cloud(cloud[:, :, 0:3])
        cloud[:, :, 0:3] = shift_point_cloud(cloud[:, :, 0:3])
        cloud = torch.Tensor(cloud)
        cloud, label, pid = cloud.float().cuda(), label.reshape((label.size(0), 1)).long().cuda(), pid.long().cuda()
        cloud = cloud.transpose(2, 1)
        out, trans_feat = net(cloud, to_categorical(label, num_classes))
        out = out.contiguous().view(-1, num_part)
        pid = pid.view(-1, 1)[:, 0]
        pred_choice = out.data.max(1)[1]

        correct = pred_choice.eq(pid.data).cpu().sum()
        mean_correct.append(correct.item() / (batch_size * pointNUM))

        loss = loss_function(out, pid, trans_feat)
        total_loss_epoch += np.float64(loss.data)

        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    avg_loss_epoch = total_loss_epoch / train_loader.__len__()
    
    # test process
    with torch.no_grad():  # no bp need
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for id in seg_classes[cat]:
                seg_label_to_cat[id] = cat

        net.eval()
        # prepare for confusion matrix
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

            correct = np.sum(cur_pred_val == pid)
            total_correct += correct
            total_seen += (cur_batch_size * pointnum)

            cur_pred_val_total.append(flatten_first_dim(cur_pred_val))
            cloud_total.append(flatten_first_dim(np.array(torch.Tensor.cpu(cloud))))
            pid_total.append(flatten_first_dim(pid))

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
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)  # 本次epoch下的mIOU

        # confusion matrix
        # flatten all data
        cur_pred_val_total = np.transpose(flatten_first_dim(cur_pred_val_total))
        cloud_total = np.transpose(flatten_first_dim(cloud_total))
        pid_total = np.transpose(flatten_first_dim(pid_total))

        Cmatrix_epoch = confusion_matrix(pid_total, cur_pred_val_total)
        # confusion_matrix_plot(cmatrix_data=Cmatrix_epoch, seg_num=num_part, savefig=False, save_dir=str(outfig_dir),
        #                       bias="epoch" + str(epoch+1) + "of" +  str(epoches) + "_" + "batchsize" + str(batch_size))

    time_end = datetime.datetime.now()
    time_span_str = str((time_end - time_start).seconds)
    print('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f' % (
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou']))
    print("epoch accuarcy: " + str(test_metrics['accuracy']))
    print("time consuming: " + time_span_str + "s")
    print("mIOU: " + str(test_metrics['instance_avg_iou']))

    # train result record
    train_result["train_accuracy"][epoch] = train_instance_acc
    train_result["accuracy"][epoch] = test_metrics["accuracy"]
    train_result["class mIOU"].append(shape_ious)
    train_result["mIOU"][epoch] = test_metrics["instance_avg_iou"]
    train_result["mIOUc"][epoch] = test_metrics["class_avg_iou"]
    train_result["loss"][epoch] = avg_loss_epoch

    # model save
    if (test_metrics['instance_avg_iou'] >= best_instance_avg_iou):
        print("epoch " + str(epoch + 1) + " will be saved.")
        savepath = str(checkpoints_dir) + '/best_model.pth'
        print('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'train_acc': train_instance_acc,
            'test_acc': test_metrics['accuracy'],
            'class_avg_iou': test_metrics['class_avg_iou'],
            'instance_avg_iou': test_metrics['instance_avg_iou'],
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        print('Saving model....')
        torch.save(state, savepath)
        # confusion matrix save
        confusion_matrix_plot(cmatrix_data=Cmatrix_epoch, seg_num=num_part, savefig=True,
                              bias="epoch" + str(epoch+1) + "of" +  str(epoches) + "_" + "batchsize" + str(batch_size))
    # optimized result output
    if test_metrics['accuracy'] > best_acc:
        best_acc = test_metrics['accuracy']
    if test_metrics['class_avg_iou'] > best_class_avg_iou:
        best_class_avg_iou = test_metrics['class_avg_iou']
    if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
        best_instance_avg_iou = test_metrics['instance_avg_iou']
    print('Best accuracy is: %.5f' % best_acc)
    print('Best class avg mIOU is: %.5f' % best_class_avg_iou)
    print('Best instance avg mIOU is: %.5f' % best_instance_avg_iou)

# plot test accuracy/loss vs epoch
plot_accuracy_loss(epochAccuracy=train_result["accuracy"], epochLoss=train_result["loss"],
                   epoches=epoches, acc_plot=True, loss_plot=True, savefig=True,
                   bias="epoch" + str(epoches) + "_" + "batchsize" + str(batch_size),
                   save_dir=str(outfig_dir))
# plot train accuracy/loss vs epoch
plot_accuracy_loss(epochAccuracy=train_result["train_accuracy"], epochLoss=train_result["loss"],
                   epoches=epoches, acc_plot=True, loss_plot=True, savefig=True,
                   bias="epoch" + str(epoches) + "_" + "batchsize" + str(batch_size) + "(train acc)",
                   save_dir=str(outfig_dir))
# plot miou vs epoch
plot_require_data(xdata=np.linspace(1, epoches, epoches), ydata=train_result["mIOU"],
                  xdatalabel="Epoch", ydatalebel="mIOU", savefig=True, save_dir=str(outfig_dir),
                  bias="epoch" + str(epoches) + "_" + "batchsize" + str(batch_size))
# plot class miou vs epoch
plot_require_data(xdata=np.linspace(1, epoches, epoches), ydata=train_result["mIOU"],
                  xdatalabel="Epoch", ydatalebel="mIOUc", savefig=True, save_dir=str(outfig_dir),
                  bias="epoch" + str(epoches) + "_" + "batchsize" + str(batch_size))