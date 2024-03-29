import os
import torch
import numpy as np
from tqdm import tqdm

# from utils.datasets.dataset import BaseDataset


class SupervisedSampler(object):
    def __init__(self,
                 dataset,
                 method: str = 'random',
                 num_frames: int = 1,
                 save_path: str = 'utils/sampling/_resources/',
                 remove_classes: list = None):
        """
        :param dataset: dataset
        :param: method: method for the frame selection
        :param num_frames: number of frames
        :param method:
        """
        self.dataset = dataset
        self.method = method
        self.num_frames = num_frames
        self.num_classes = self.dataset.num_classes
        self.save_path = save_path
        if 'kitti' in self.dataset.name.lower():
            self.dataset_name = 'semantickitti'
        elif 'nuscenes' in self.dataset.name.lower():
            self.dataset_name = 'nuscenes'
        else:
            self.dataset_name = 'semanticposs'
        
        self.file_name = os.path.join(self.save_path, self.dataset_name + '_' + self.method + '.npy')

        self.remove_classes = remove_classes

        print(f'SSDA FILENAME {self.file_name}')

        self.best_idx = self.sample_frames()

        print(f'--> SSDA FRAME {self.best_idx}')

    def sample_frames(self):

        if self.method == 'random':
            best_idx = np.random.randint(0, len(self.dataset))
        elif self.method == 'best':
            if not os.path.isfile(self.file_name):
                best_idx = self.find_all_classes_frames()
            else:
                best_idx = np.load(self.file_name)

        elif self.method == 'custom':
            if self.dataset_name == 'semantickitti':
                # following synlidar 000848 in seq 06 and 000940 in seq 02
                path0 = '06/velodyne/000848.bin'
                path1 = '02/velodyne/000940.bin'
                best_idx = []
                for idx, p in enumerate(self.dataset.pcd_path):
                    if path0 in p or path1 in p:
                        print(f'---> FOUND: {p}')
                        best_idx.append(idx)
                print(f'---> FOUND: {best_idx}')
            elif self.dataset_name == 'semanticposs':
                # following synlidar 000172 in seq 02
                path = '02/velodyne/000172.bin'
                for idx, p in enumerate(self.dataset.pcd_path):
                    if path in p:
                        print(f'---> FOUND: {p}')
                        best_idx = idx
                        break
            elif self.dataset_name == 'nuscenes':
                path = 'n015-2018-07-24-11-13-19+0800__LIDAR_TOP__1532402013197655.pcd.bin'
                for idx, p in enumerate(self.dataset.pcd_path):
                    if path in p:
                        print(f'---> FOUND: {p}')
                        best_idx = idx
                        break
            else:
                raise NotImplementedError

        elif self.method == 'long_tail':
            raise NotImplementedError
        else:
            raise NotImplementedError

        return best_idx

    def find_all_classes_frames(self):
        frames = {}

        for i in tqdm(range(len(self.dataset)), desc='Frame selection'):
            data_tmp = self.dataset.__getitem__(i)
            lbl_tmp = data_tmp['labels']
            classes = np.unique(lbl_tmp)
            classes = classes[classes != -1]
            frames[i] = len(classes)

        idx, nums = zip(*frames.items())

        idx = np.asarray(idx)
        nums = np.asarray(nums)

        max_classes = np.argmax(nums)
        selected_idx = idx[max_classes]

        os.makedirs(self.save_path, exist_ok=True)
        np.save(os.path.join(self.file_name), selected_idx)

        return selected_idx

    def remove_class(self, data, rc):
        pts = data["coordinates"]
        feats = data["features"]
        lbl = data["labels"]

        cls_idx = torch.logical_not(lbl == rc).bool()
        data["coordinates"] = pts[cls_idx]
        data["features"] = feats[cls_idx]
        data["labels"] = lbl[cls_idx]

        return data

    def get_frame(self):
        """
        :return: a loaded frame, imitates __getitem__ in standard datasets
        """
        if self.num_frames == 1:
            data = self.dataset.__getitem__(self.best_idx)
        else:
            i = np.random.choice(self.best_idx)
            data = self.dataset.__getitem__(i)

        if self.remove_classes is not None:
            for rc in self.remove_classes:
                data = self.remove_class(data, rc=rc)

        return data

