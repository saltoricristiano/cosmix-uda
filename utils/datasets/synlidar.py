import os
import torch
import yaml
import numpy as np

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class SynLiDARDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = '/data/csaltori/SynLiDAR/',
                 mapping_path: str = '_resources/synlidar_semantickitti.yaml',
                 weights_path: str = None,
                 voxel_size: float = 0.05,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 sub_num: int = 80000,
                 device: str = None,
                 num_classes: int = 7,
                 ignore_label: int = None):

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)

        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path)

        self.name = 'SynLiDARDataset'
        if self.version == 'full':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'mini':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'sequential':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        else:
            raise NotImplementedError

        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []
        self.selected = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        for sequence, frames in self.split[self.phase].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        print(f'--> Selected {len(self.pcd_path)} for {self.phase}')

    def get_splits(self):
        if self.version == 'full':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar.pkl')
        elif self.version == 'mini':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError

        if not os.path.isfile(split_path):
            self.split = {'train': {s: [] for s in self.sequences},
                          'validation': {s: [] for s in self.sequences}}
            if self.version != 'sequential':
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    valid_frames = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            valid_frames.append(v)
                    if self.version == 'full':
                        train_selected = np.random.choice(valid_frames, int(num_frames/10), replace=False)
                    else:
                        train_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    for t in train_selected:
                        valid_frames.remove(t)

                    validation_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
            else:
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    total_for_sequence = int(num_frames/10)
                    print('--> TOTAL:', total_for_sequence)
                    train_selected = []
                    validation_selected = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            if len(train_selected) == 0:
                                train_selected.append(v)
                                last_added = v
                            elif len(train_selected) < total_for_sequence and (v-last_added) >= 5:
                                train_selected.append(v)
                                last_added = v
                                print(last_added)
                            else:
                                validation_selected.append(v)

                    validation_selected = np.random.choice(validation_selected, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)

        else:
            self.split = torch.load(split_path)
            print('SEQUENCES', self.split.keys())
            print('TRAIN SEQUENCES', self.split['train'].keys())

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        sampled_idx = np.arange(points.shape[0])
        if self.phase == 'train' and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(points,
                                                                               colors,
                                                                               labels=labels,
                                                                               ignore_label=vox_ign_label,
                                                                               quantization_size=self.voxel_size,
                                                                               return_index=True)

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i)}

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(label_tmp, dtype=np.uint32)
        label = label.reshape((-1))
        label = self.remap_lut_val[label].astype(np.int32)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": i}

