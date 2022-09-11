import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import os
import json
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import normalSpeed

from PIL import Image
import numpy.ma as ma
import scipy.io as scio

import math

from time import perf_counter

def load_data(partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob('./data/modelnet40_ply_hdf5_2048/ply_data_%s*.h5' % partition):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=8./9., high=9./8., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


# =========== ModelNet40 =================
class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition  # Here the new given partition will cover the 'train'

    def __getitem__(self, item):  # indice of the pts or label
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = pc_normalize(pointcloud)  # you can try to add it or not to train our model
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)  # shuffle the order of pts
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# =========== ShapeNet Part =================
class PartNormalDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False):
        self.npoints = npoints
        self.root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            normal = data[:, 3:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.normalize:
            point_set = pc_normalize(point_set)

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice, :]

        return point_set, cls, seg, normal

    def __len__(self):
        return len(self.datapath)

#depth should be in millimeters
def compute_normals(depth, fx, fy, k_size=5, distance_threshold=2000, difference_threshold=20, point_into_surface=False):
    normals = normalSpeed.depth_normal(depth, fx, fy, k_size, distance_threshold, difference_threshold, point_into_surface)
    return normals

_EPS = 1e-5

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> np.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(np.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                        np.cos(t1)*r1, np.sin(t2)*r2])

def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

#YCB version
class YCBDataset(Dataset):
    def __init__(self, npoints=2500, split='train', normalize=False, return_project_data=False, crop_object=False):
        self.load_data(split)
        self.npoints = npoints
        self.split = split

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
        self.normalize = normalize

        self.return_project_data = return_project_data
        self.crop_object = crop_object

    def load_data(self, split):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        self.root = os.path.join(DATA_DIR, "YCB_Video_Dataset")

        if split == 'train':
            self.path = os.path.join(DATA_DIR, "dataset_config", "train_data_list.txt")
            self.add_noise = True
        elif split == 'test':
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

        print("{0} loaded: size: {1}".format(split, len(self.list)))

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

        img = np.array(img)[:, :, :3]
        orig_img = np.copy(img)

        if self.list[index][:8] == 'data_syn':
            img, depth = self.add_real_back(img, label, depth, mask_depth)

        #basically, half the time, perform a crop
        #we need to train the network to understand the individual objects
        if (self.split == "train" and np.random.rand() > 0.6) or self.crop_object:

            while True:
                obj = meta['cls_indexes'].flatten().astype(np.int32)

                idx = np.random.choice(len(obj), 1)
                obj_idx = obj[idx]
                mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))

                if len(mask_label.nonzero()[0]) <= self.minimum_num_pt:
                    continue
            
                rmin, rmax, cmin, cmax = get_bbox(mask_label)

                h, w, _= img.shape
                rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                img = img[rmin:rmax, cmin:cmax]
                depth = depth[rmin:rmax, cmin:cmax]
                xmap = self.xmap[rmin:rmax, cmin:cmax]
                ymap = self.ymap[rmin:rmax, cmin:cmax]
                label = label[rmin:rmax, cmin:cmax]

                break

        else:
            choose = mask.flatten().nonzero()[0]
            xmap = self.xmap
            ymap = self.ymap

        if len(choose) == 0:
            return {}

        if len(choose) > self.npoints:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.npoints] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.npoints - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
        label_masked = label.flatten()[choose][:, np.newaxis].astype(np.int64)
        choose = np.array([choose])

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        if self.normalize:
            cloud = pc_normalize(cloud)

        #zero mean the point cloud
        cloud_mean = np.mean(cloud, axis=0)
        cloud -= cloud_mean

        if self.add_noise:
            cloud = translate_pointcloud(cloud)
        
        noise_trans = 0.005

        if self.add_noise:
            #random jitter
            add_t = np.random.uniform(-noise_trans, noise_trans, (cloud.shape[0], 3))
            cloud = np.add(cloud, add_t)

            #random rotation
            random_rot_mat = random_rotation_matrix()[:3,:3]
            cloud = cloud @ random_rot_mat.T

        depth_mm = (depth * (1000 / cam_scale)).astype(np.uint16)
        normals = compute_normals(depth_mm, cam_fx, cam_fy)
        normals_masked = normals.reshape((-1, 3))[choose].astype(np.float32).squeeze(0)

        if self.return_project_data:
            project_data = {}
            project_data["intr"] = (cam_cx, cam_cy, cam_fx, cam_fy)
            project_data["cloud_mean"] = cloud_mean
            project_data["img"] = orig_img
            return cloud, np.array([0]), label_masked, normals_masked, project_data
        else:
            #no overall class for pointcloud
            return cloud, np.array([0]), label_masked, normals_masked

    def __getitem__(self, index):

        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        x =  self.get_item(index, img, depth, label, meta)
        return x

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    train = YCBDataset(npoints=2048, split='train', normalize=False)
    test = YCBDataset(npoints=2048, split='test', normalize=False)
    for data, label, _, _ in train:
        print(data.shape)
        print(label.shape)
