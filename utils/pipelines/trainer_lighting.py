import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from utils.losses import CELoss, DICELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d


class PLTTrainer(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 lr=1e-3,
                 batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None):

        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=None)

        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)

        elif criterion == 'SoftDICELoss':
            if self.num_classes == 19:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label, is_kitti=True)
            else:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)

        else:
            raise NotImplementedError

        self.ignore_label = self.training_dataset.ignore_label
        self.validation_phases = ['source_validation', 'target_validation']

        self.save_hyperparameters(ignore='model')

    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        loss = self.criterion(out, labels)

        _, preds = out.max(1)
        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join('training', p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict['training/loss'] = loss
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])
        results_dict['training/lr'] = self.trainer.optimizers[0].param_groups[0]["lr"]
        results_dict['training/epoch'] = self.current_epoch

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.batch_size
            )
        return loss

    def validation_step(self, batch, batch_idx):
        phase = 'source_validation'

        # input batch
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F.cpu()

        labels = batch['labels'].long().cpu()

        loss = self.criterion(out, labels)

        soft_pseudo = F.softmax(out, dim=-1)

        conf, preds = soft_pseudo.max(1)

        iou_tmp = jaccard_score(preds.detach().numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-1)

        present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.val_batch_size,
                add_dataloader_idx=False)
        return results_dict

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")

            else:
                raise NotImplementedError

            return [optimizer], [scheduler]


class PLTTester(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 criterion='CELoss',
                 dataset=None,
                 clear_cache_int=2,
                 num_classes=19,
                 checkpoint_path=None,
                 save_predictions=False,
                 save_folder=None):

        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.ignore_label = self.dataset.ignore_label

        # if criterion == 'CELoss':
        #     self.criterion = CELoss(ignore_label=self.ignore_label,
        #                             weight=None)
        # else:
        #     raise NotImplementedError

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        phase = 'test'
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        # loss = self.criterion(out, labels)
        conf = F.softmax(out, dim=-1)
        preds = conf.max(dim=-1).indices
        conf = conf.max(dim=-1).values

        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-0.1)

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        iou_tmp = torch.from_numpy(iou_tmp)

        iou = -torch.ones_like(iou_tmp)
        iou[present_labels] = iou_tmp[present_labels]

        if self.save_predictions:
            coords = batch["coordinates"].cpu()
            labels = batch["labels"].cpu()
            preds = preds.cpu()
            conf = conf.cpu()

            batch_size = torch.unique(coords[:, 0]).max() + 1
            sample_idx = batch["idx"]
            for b in range(batch_size.int()):
                s_idx = int(sample_idx[b].item())
                b_idx = coords[:, 0] == b
                points = coords[b_idx, 1:]
                p = preds[b_idx]
                c = conf[b_idx]
                l = labels[b_idx]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(self.dataset.color_map[p+1])

                iou_tmp = jaccard_score(p.cpu().numpy(), l.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=-0.1)

                present_labels, _ = np.unique(l.cpu().numpy(), return_counts=True)
                present_labels = present_labels[present_labels != self.ignore_label]
                iou_tmp = np.nanmean(iou_tmp[present_labels]) * 100

                os.makedirs(os.path.join(self.save_folder, 'preds'), exist_ok=True)
                os.makedirs(os.path.join(self.save_folder, 'labels'), exist_ok=True)
                os.makedirs(os.path.join(self.save_folder, 'pseudo'), exist_ok=True)

                o3d.io.write_point_cloud(os.path.join(self.save_folder, 'preds', f'{s_idx}_{int(iou_tmp)}.ply'), pcd)

                pcd.colors = o3d.utility.Vector3dVector(self.dataset.color_map[l+1])

                o3d.io.write_point_cloud(os.path.join(self.save_folder, 'labels', f'{s_idx}.ply'), pcd)

                valid_pseudo = c > 0.85
                p[torch.logical_not(valid_pseudo)] = -1

                pcd.colors = o3d.utility.Vector3dVector(self.dataset.color_map[p+1])

                o3d.io.write_point_cloud(os.path.join(self.save_folder, 'pseudo', f'{s_idx}.ply'), pcd)


        # return {'iou': iou, 'loss': loss.cpu()}
        return {'iou': iou}

    def test_epoch_end(self, outputs):
        mean_iou = []
        # mean_loss = []

        for return_dict in outputs:
            iou_tmp = return_dict['iou']
            # loss_tmp = return_dict['loss']

            nan_idx = iou_tmp == -1
            iou_tmp[nan_idx] = float('nan')
            mean_iou.append(iou_tmp.unsqueeze(0))
            # mean_loss.append(loss_tmp)

        mean_iou = torch.cat(mean_iou, dim=0).numpy()

        per_class_iou = np.nanmean(mean_iou, axis=0) * 100
        # loss = np.mean(mean_loss)

        results = {'iou': np.mean(per_class_iou)}

        for c in range(per_class_iou.shape[0]):
            class_name = self.dataset.class2names[c]
            results[class_name] = per_class_iou[c]

        os.makedirs(os.path.join(self.trainer.weights_save_path, 'results'), exist_ok=True)
        csv_columns = list(results.keys())

        _, ckpt_name = os.path.split(self.checkpoint_path)
        ckpt_name = ckpt_name[:-5]

        csv_file = os.path.join(self.trainer.weights_save_path, 'results', ckpt_name+'_test.csv')
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(results)

