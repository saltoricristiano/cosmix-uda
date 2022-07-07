import os
import torch
import yaml
import numpy as np
import tqdm

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class SemanticPOSSDataset(BaseDataset):
    def __init__(self,
                 version='full',
                 phase='train',
                 dataset_path='/data/csaltori/SemanticPOSS/sequences',
                 mapping_path='_resources/semanticposs.yaml',
                 weights_path=None,
                 voxel_size=0.05,
                 use_intensity=False,
                 augment_data=False,
                 sub_num=50000,
                 device=None,
                 num_classes=7,
                 ignore_label=None):

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

        if self.version == 'full':
            self.split = {'train': ['00', '01', '02', '04', '05'],
                          'validation': ['03']}
        elif self.version == 'mini':
            self.split = {'train': ['00', '05'],
                          'validation': ['03']}
        else:
            raise NotImplementedError

        self.name = 'SemanticPOSSDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val

        for sequence in self.split[self.phase]:
            num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))

            for f in np.arange(num_frames):
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(f):06d}.label')

                if os.path.exists(pcd_path) and os.path.exists(label_path):
                    self.pcd_path.append(pcd_path)
                    self.label_path.append(label_path)

        self.color_map = np.array([(255, 255, 255),  # unlabelled
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # rider
                                    (25, 25, 255),  # car
                                    (107, 98, 56),  # trunk
                                    (157, 234, 50),  # plants
                                    (173, 23, 121),  # traffic-sign
                                    (83, 93, 130),  # pole
                                    (23, 173, 148),  # garbage-can
                                    (233, 166, 250),  # building
                                    (173, 23, 0),  # traffic-cone
                                    (255, 214, 251),  # fence
                                    (187, 0, 255),  # bicycle
                                    (164, 173, 104)])/255.  # other-ground

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_poss(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
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
        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        return {"coordinates": quantized_coords,
                "features": torch.from_numpy(feats),
                "labels": torch.from_numpy(labels),
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i)}

    def load_label_poss(self, label_path):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1)
        for l in tqdm.tqdm(range(len(self.label_path)), desc='Loading weights', leave=True):
            label_tmp = self.label_path[l]
            label = self.load_label_poss(label_tmp)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_poss(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
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
