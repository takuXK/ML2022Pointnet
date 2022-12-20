import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


main_path = "..\\data\\hdf5_data\\"

def get_data(train_txt_path, valid_txt_path, train=True):
    data_txt_path = train_txt_path if train else valid_txt_path

    with open(data_txt_path, "r") as f:
        txt = f.read()
    clouds_li = []
    labels_li = []
    partIDs_li = []
    for file_name in txt.split():
        h5 = h5py.File(main_path + file_name)
        pts = h5["data"][:]
        lbl = h5["label"][:]
        pid = h5["pid"][:]
        clouds_li.append(torch.Tensor(pts))
        labels_li.append(torch.Tensor(lbl))
        partIDs_li.append(torch.Tensor(pid))
    clouds = torch.cat(clouds_li)
    labels = torch.cat(labels_li)
    partIDs = torch.cat(partIDs_li)
    return clouds, labels.long().squeeze(), partIDs


class PointDataSet(Dataset):
    def __init__(self, train_txt_path, valid_txt_path, train=True):
        clouds, labels, pids = get_data(train_txt_path, valid_txt_path, train=train)

        self.x_data = clouds
        self.y_data = labels
        self.z_data = pids

        self.lenth = clouds.size(0)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.lenth


def get_dataLoader(train_txt_path, valid_txt_path, train=True, batch_size=2):
    point_data_set = PointDataSet(train_txt_path, valid_txt_path, train=train)
    data_loader = DataLoader(dataset=point_data_set, batch_size=batch_size, shuffle=train)
    return data_loader