import os
import glob
from PIL import Image
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import numpy.ma as ma
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class YCB(Dataset):
    def __init__(self, num_points, partition='train'):
        self.load_data(partition)
        self.num_points = num_points
        self.partition = partition

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])       

        self.minimum_num_pt = 50

    def load_data(self, partition):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        self.root = os.path.join(DATA_DIR, "YCB_Video_Dataset")

        if partition == 'train':
            self.path = os.path.join(DATA_DIR, "dataset_config", "train_data_list.txt")
            self.add_noise = True
        elif partition == 'test':
            self.path = os.path.join(DATA_DIR, "dataset_config", "test_data_list.txt")
            self.add_noise = False #only add noise to training samples

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        print("{0} loaded: size: {1}".format(partition, len(self.list)))

    def real_gen(self):
        n = len(self.real)
        idx = np.random.randint(0, n)
        item = self.real[idx]
        return item

    def rand_range(self, lo, hi):
        return np.random.rand()*(hi-lo)+lo

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def get_item(self, index, img, depth, label, meta, return_intr=False):

        cam_scale = meta['factor_depth'][0][0]

        # if self.cfg.fill_depth:
        #     depth = fill_missing(depth, cam_scale, 1)

        # if self.cfg.blur_depth:
        #     depth = cv2.GaussianBlur(depth,(3,3),cv2.BORDER_DEFAULT)

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_depth
        if len(mask.nonzero()[0]) <= self.minimum_num_pt:
            return {}

        choose = mask.flatten().nonzero()[0]

        if len(choose) == 0:
            return {}

        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        img = np.array(img)[:, :, :3]

        if self.list[index][:8] == 'data_syn':
            img, depth = self.add_real_back(img, label, depth, mask_depth)

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        label_masked = label.flatten()[choose][:, np.newaxis].astype(np.int64)
        choose = np.array([choose])

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        if self.add_noise:
            cloud = translate_pointcloud(cloud)

        return cloud, label_masked

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        return self.get_item(index, img, depth, label, meta)

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    train = YCB(1024)
    test = YCB(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(YCB(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = YCB(partition='train', num_points=1024)
    test_set = YCB(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
