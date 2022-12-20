import numpy as np
import random
import h5py
import os
import tqdm

def CloudPointRandomSample(originPointData, originLabel, numSample=1024, normalization=True):
    numPoints = len(originLabel)

    if numPoints < numSample:
        raise ValueError("the number of points is less than the number you want to sample.")
    elif numPoints == numSample:
        print("the number of points is equals to the number you want to sample and the original data will output")
        samplePointData = originPointData
        sampleLabel = originLabel
    else:
        # 随机采样index
        sampleIndex = random.sample(list(range(numPoints)), numSample)
        # 将index映射到点云数据样本上
        samplePointData = np.array([originPointData[i] for i in sampleIndex])
        sampleLabel = np.array([originLabel[i] for i in sampleIndex])

    finalData = {"point": samplePointData, "label": sampleLabel}

    # 当需要归一化数据时
    if normalization:
        normPointData = samplePointData - np.mean(samplePointData, axis=0)
        normPointData = normPointData / np.max(np.linalg.norm(normPointData, axis=1))

        finalData = {"point": normPointData, "label": sampleLabel}

    return finalData


def get_txtData(filename, seg_class, random_sample=False, num_sample=512):
    load_data = np.loadtxt(filename)
    data = np.array([x for x in load_data]).astype(np.float32)

    point = data[:, 0:3]
    pointlabel = data[:, -1]
    classlabel = np.array([seg_class])

    if random_sample:
        temp_point = point
        temp_pointlabel = pointlabel
        point = CloudPointRandomSample(temp_point, temp_pointlabel, num_sample, False)["point"]
        pointlabel = CloudPointRandomSample(temp_point, temp_pointlabel, num_sample, False)["label"]

    return point, pointlabel, classlabel

def get_multi_txtData(filepath, seg_class_list, random_sample=True, num_sample=512):
    filelist = os.listdir(filepath)
    if len(seg_class_list) < len(filelist):
        raise ValueError("the length of 'seg_class_list' is not enough.")
    points, pointlabels, classlabels = [], [], []
    class_index = 0
    for filename in tqdm.tqdm(filelist):
        txtfilename = os.path.join(filepath, filename)
        point, pointlabel, classlabel = get_txtData(filename=txtfilename, seg_class=seg_class_list[class_index],
                                                    random_sample=random_sample, num_sample=num_sample)
        points.append(point)
        pointlabels.append(pointlabel)
        classlabels.append(classlabel)
        class_index += 1

    # reshape the data for h5 file save
    points_shape = points[0].shape
    points = np.reshape(points, (len(points), points_shape[0], points_shape[1]))
    pointlabels_shape = pointlabels[0].shape
    pointlabels = np.reshape(pointlabels, (len(pointlabels), pointlabels_shape[0]))
    classlabels = np.reshape(classlabels, (len(classlabels), 1))

    return points, pointlabels, classlabels


def write_h5file(data, h5filename):
    h5file = h5py.File(h5filename, 'w')

    try:
        h5file["data"] = data["point"]
        h5file["label"] = data["classlabel"]
        h5file["pid"] = data["pointlabel"]
    except KeyError:
        print("the key is invalid.")

    h5file.close()




if __name__ == "__main__":
    # filename = "..\\data\\shapenetcore_partanno_segmentation_benchmark_v0_normal\\02691156\\1a04e3eab45ca15dd86060f189eb133.txt"
    h5filename = "..\\data\\hdf5_data\\test.h5"
    filepath = "..\\Pointnet2022ML\\testpath\\"

    class_list = {
        'Airplane': 0,
        'Bag':1,
        'Cap': 2,
        'Car': 3,
        'Chair': 4,
        'Earphone': 5,
        'Guitar': 6,
        'Knife': 7,
        'Lamp': 8,
        'Laptop': 9,
        'Motorbike': 10,
        'Mug': 11,
        'Pistol': 12,
        'Rocket': 13,
        'Skateboard': 14,
        'Table': 15
    }

    seg_cls_list = []
    for filename in os.listdir(filepath):
        classname = filename.split('.')[0]
        seg_cls_list.append(class_list[classname])

    # point, pointlabel, classlabel = get_txtData(filename, seg_class=1)

    points, pointlabels, classlabels = get_multi_txtData(filepath, seg_class_list=seg_cls_list, num_sample=2048)
    data = {
        "point": points,
        "classlabel": classlabels,
        "pointlabel": pointlabels
    }
    write_h5file(data=data, h5filename=h5filename)

    h5 = h5py.File(h5filename)
    pts = h5["data"][:]
    lbl = h5["label"][:]
    pid = h5["pid"][:]