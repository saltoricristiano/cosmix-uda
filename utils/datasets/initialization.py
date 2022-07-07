import os
import numpy as np
from utils.datasets.dataset import BaseDataset
from utils.datasets.concat_dataset import ConcatDataset
from utils.datasets.semantickitti import SemanticKITTIDataset
from utils.datasets.semanticposs import SemanticPOSSDataset
from utils.datasets.synlidar import SynLiDARDataset


synlidar2kitti = np.array(['car', 'bicycle', 'motorcycle',  'truck', 'other-vehicle', 'person',
                           'bicyclist', 'motorcyclist',
                           'road', 'parking', 'sidewalk', 'other-ground',
                           'building', 'fence', 'vegetation', 'trunk',
                           'terrain', 'pole', 'traffic-sign'])

synlidar2poss = np.array(['person', 'rider', 'car', 'trunk',
                          'plants', 'traffic-sign', 'pole', 'garbage-can',
                          'building', 'cone', 'fence', 'bike', 'ground'])


synlidar2kitti_color = np.array([(255, 255, 255),  # unlabelled
                                    (25, 25, 255),  # car
                                    (187, 0, 255),  # bicycle
                                    (187, 50, 255),  # motorcycle
                                    (0, 247, 255),  # truck
                                    (50, 162, 168),  # other-vehicle
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # bicyclist
                                    (255, 196, 0),  # motorcyclist
                                    (0, 0, 0),  # road
                                    (148, 148, 148),  # parking
                                    (255, 20, 60),  # sidewalk
                                    (164, 173, 104),  # other-ground
                                    (233, 166, 250),  # building
                                    (255, 214, 251),  # fence
                                    (157, 234, 50),  # vegetation
                                    (107, 98, 56),  # trunk
                                    (78, 72, 44),  # terrain
                                    (83, 93, 130),  # pole
                                    (173, 23, 121)])/255.   # traffic-sign


synlidar2poss_color = np.array([(255, 255, 255),  # unlabelled
                                (250, 178, 50),  # person
                                (255, 196, 0),  # rider
                                (25, 25, 255),  # car
                                (107, 98, 56),  # trunk
                                (157, 234, 50),  # plants
                                (233, 166, 250),  # building
                                (255, 214, 251),  # fence
                                (0, 0, 0),   # ground
                                (173, 23, 121),  # traffic-sign
                                (83, 93, 130)])/255.  # ground


def get_dataset(dataset_name: str,
                dataset_path: str,
                voxel_size: float = 0.02,
                sub_num: int = 50000,
                augment_data: bool = False,
                version: str = 'mini',
                num_classes: int = 7,
                ignore_label: int = -1,
                mapping_path: str = None,
                target_name: str = None,
                weights_path: str = None) -> (BaseDataset, BaseDataset, BaseDataset):

    '''
        :param dataset_name: name of the dataset
        :param dataset_path: absolute path to data
        :param voxel_size: voxel size for voxelization
        :param sub_num: number of points sampled
        :param augment_data: if to augment data
        :param version: mini/full dataset
        :param num_classes: number of classes considered
        :param ignore_label: label to ignore
        :param mapping_path: path to mapping files for labels
        :param target_name: name of the target dataset used in the experiments
        :param weights_path: path to weights of the dataset
        :return:
    '''

    if dataset_name == 'SemanticKITTI':

        training_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                mapping_path=mapping_path,
                                                version=version,
                                                phase='train',
                                                voxel_size=voxel_size,
                                                augment_data=augment_data,
                                                sub_num=sub_num,
                                                num_classes=num_classes,
                                                ignore_label=ignore_label,
                                                weights_path=weights_path)
        validation_dataset = SemanticKITTIDataset(dataset_path=dataset_path,
                                                  mapping_path=mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  augment_data=False,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label,
                                                  weights_path=weights_path)
        target_dataset = None

        training_dataset.class2names = synlidar2kitti
        validation_dataset.class2names = synlidar2kitti
        training_dataset.color_map = synlidar2kitti_color
        validation_dataset.color_map = synlidar2kitti_color

    elif dataset_name == 'SynLiDAR':

        training_dataset = SynLiDARDataset(dataset_path=dataset_path,
                                           version=version,
                                           phase='train',
                                           voxel_size=voxel_size,
                                           augment_data=augment_data,
                                           num_classes=num_classes,
                                           ignore_label=ignore_label,
                                           mapping_path=mapping_path,
                                           weights_path=weights_path)

        validation_dataset = SynLiDARDataset(dataset_path=dataset_path,
                                             version=version,
                                             phase='validation',
                                             voxel_size=voxel_size,
                                             augment_data=False,
                                             num_classes=num_classes,
                                             ignore_label=ignore_label,
                                             mapping_path=mapping_path,
                                             weights_path=weights_path)

        if target_name == 'SemanticKITTI':
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = os.path.join(main_path, 'SemanticKITTI/data/sequences/')

            target_mapping_path = '_resources/semantic-kitti.yaml'
            target_dataset = SemanticKITTIDataset(dataset_path=target_dataset_path,
                                                  mapping_path=target_mapping_path,
                                                  version=version,
                                                  phase='validation',
                                                  voxel_size=voxel_size,
                                                  augment_data=False,
                                                  num_classes=num_classes,
                                                  ignore_label=ignore_label)

            training_dataset.class2names = synlidar2kitti
            validation_dataset.class2names = synlidar2kitti
            target_dataset.class2names = synlidar2kitti

            training_dataset.color_map = synlidar2kitti_color
            validation_dataset.color_map = synlidar2kitti_color
            target_dataset.color_map = synlidar2kitti_color

        elif target_name == 'SemanticPOSS':
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = os.path.join(main_path, 'SemanticPOSS/sequences/')
            target_mapping_path = '_resources/semanticposs.yaml'
            target_dataset = SemanticPOSSDataset(dataset_path=target_dataset_path,
                                                 mapping_path=target_mapping_path,
                                                 version=version,
                                                 phase='validation',
                                                 voxel_size=voxel_size,
                                                 augment_data=False,
                                                 num_classes=num_classes,
                                                 ignore_label=ignore_label)

            training_dataset.class2names = synlidar2poss
            validation_dataset.class2names = synlidar2poss
            target_dataset.class2names = synlidar2poss

            training_dataset.color_map = synlidar2poss_color
            validation_dataset.color_map = synlidar2poss_color
            target_dataset.color_map = synlidar2poss_color

        else:
            raise NotImplementedError

    elif dataset_name == 'SemanticPOSS':

        training_dataset = SemanticPOSSDataset(dataset_path=dataset_path,
                                               mapping_path=mapping_path,
                                               version=version,
                                               phase='train',
                                               voxel_size=voxel_size,
                                               augment_data=augment_data,
                                               sub_num=sub_num,
                                               num_classes=num_classes,
                                               ignore_label=ignore_label,
                                               weights_path=weights_path)
        validation_dataset = SemanticPOSSDataset(dataset_path=dataset_path,
                                                 mapping_path=mapping_path,
                                                 version=version,
                                                 phase='validation',
                                                 voxel_size=voxel_size,
                                                 augment_data=False,
                                                 num_classes=num_classes,
                                                 ignore_label=ignore_label,
                                                 weights_path=weights_path)

        target_dataset = None
        training_dataset.class2names = synlidar2poss
        validation_dataset.class2names = synlidar2poss
        training_dataset.color_map = synlidar2poss_color
        validation_dataset.color_map = synlidar2poss_color

    else:
        raise NotImplementedError

    return training_dataset, validation_dataset, target_dataset


def get_concat_dataset(source_dataset,
                       target_dataset,
                       augment_data=False,
                       augment_mask_data=False,
                       remove_overlap=False):

    return ConcatDataset(source_dataset=source_dataset,
                         target_dataset=target_dataset,
                         augment_data=augment_data,
                         augment_mask_data=augment_mask_data,
                         remove_overlap=remove_overlap)
