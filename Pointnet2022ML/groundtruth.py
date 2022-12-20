from data_visualization import CloudPointVisualization
import os
from txtTOh5file import get_txtData
import numpy as np

main_path = "..\\Pointnet2022ML\\testpath\\"

savepath = "..\\Pointnet2022ML\\log\\reconstruct_fig"

for filename in os.listdir(main_path):
    classname = filename.split('.')[0]
    points, labels, _ = get_txtData(main_path + filename, 0)

    CloudPointVisualization(pointData=points, label=labels.astype(np.int32),
                            title=classname, bias="groundtruth", savefig=True, save_dir=str(savepath))