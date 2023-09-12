import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib
import glob
join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

class Brats_loadall_nii(Dataset): #加载所有的数据；
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        # data_file_path = os.path.join(root, train_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()

        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol', '*_vol.npy')) #获取指定文件内的所有符合该格式的文件名；
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])#按照病例的编号进行排序；
        print('###############', len(patients_dir))
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients) #创建一个数组，其中是对应病例的序号；
        np.random.seed(0)
        np.random.shuffle(pid_idx) #随机进行序号打乱；
        n_fold_list = np.split(pid_idx, 3) #将数据进行划分成3份；

        volpaths = []
        for i, fold in enumerate(n_fold_list): #这里的n_fold_list一共有三份，下面取第2、3份作为训练数据；
            if i != 0: #当o=1或者2的时候存为训练数据；
                for idx in fold: #fold是每一个fold的数据；
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths] #datalist中保存了训练病例的名称；
        '''Yao'''

        self.volpaths = volpaths #训练数据的病例路径；
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist  #训练数据的病例名称
        self.num_cls = num_cls #类别数量；
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3]) #表示所有的模态；

    def __getitem__(self, index):

        volpath = self.volpaths[index] #根据索引获取数据的路径；
        name = self.names[index] #根据索引获取数据的名称；
        
        x = np.load(volpath) #加载npy格式的数据；
        segpath = volpath.replace('vol', 'seg')  #将vol替换成seg，也就是想要加载对应的seg真值标签；
        y = np.load(segpath) #加载真值标签数据；
        #根据预先处理数据的代码可知加载进来的数据维度为[H,W,D,4]
        x, y = x[None, ...], y[None, ...] #加载进来的数据在第0维度上扩充一维；  图像数据[[H,W,D,4]] 真值[[H,W,D]]

        x,y = self.transforms([x, y]) #同时对图像和对应的标签进行transform变换；

        #ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y) #获得真值标签的形状尺寸；
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]#将真值标签进行one-hot处理；
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1)) #这里就是对于立方体的每一个位置都有一个one-hot的向量表示对应的标签；
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))#将类别数改为通道数位置；

        x = x[:, self.modal_ind, :, :, :] #根据已经有的模态进行输入数据的模态选择；

        x = torch.squeeze(torch.from_numpy(x), dim=0) #将数据转化为tensor类型并且缩减维度；[channels,Height,Width,Depth]
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1) #从mask所有情况中随机选择一个
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name  #返回当前加载的一组数据（原始四种模态的数据x、对应one-hot后的标签yo、对应随机进行模态缺失的情况mask、当前病例的名称name）

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        # data_file_path = os.path.join(root, test_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()
        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 3)

        volpaths = []
        for i, fold in enumerate(n_fold_list):
            if i == 0: #使用第0折数据进行测试；
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]
        '''Yao'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
