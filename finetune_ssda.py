import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

import utils.models as models
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.collation import CollateFN, CollateSSDA
from utils.pipelines import PLTTrainer
from utils.sampling.ssda import SupervisedSampler

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")


def load_model(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt[k]
            del ckpt[k]
        return state

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model


def train(config):

    def get_dataloader(dataset, batch_size, collate_fn, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        source_mapping_path = config.source_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for source!'):
        source_mapping_path = None

    try:
        target_mapping_path = config.target_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for target!'):
        target_mapping_path = None

    source_training_dataset, source_validation_dataset, _ = get_dataset(dataset_name=config.source_dataset.name,
                                                                        dataset_path=config.source_dataset.dataset_path,
                                                                        voxel_size=config.source_dataset.voxel_size,
                                                                        augment_data=config.source_dataset.augment_data,
                                                                        version=config.source_dataset.version,
                                                                        sub_num=config.source_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.source_dataset.ignore_label,
                                                                        mapping_path=source_mapping_path,
                                                                        target_name=config.target_dataset.name,
                                                                        weights_path=config.source_dataset.weights_path)

    target_training_dataset, target_validation_dataset, _ = get_dataset(dataset_name=config.target_dataset.name,
                                                                        dataset_path=config.target_dataset.dataset_path,
                                                                        voxel_size=config.target_dataset.voxel_size,
                                                                        augment_data=config.target_dataset.augment_data,
                                                                        version=config.target_dataset.version,
                                                                        sub_num=config.target_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.target_dataset.ignore_label,
                                                                        mapping_path=target_mapping_path)
    
    ssda_sampler = SupervisedSampler(dataset=target_training_dataset,
                                     method=config.adaptation.ssda_sampler.method,
                                     num_frames=config.adaptation.ssda_sampler.num_frames)

    collation = CollateFN()
    training_dataloader = get_dataloader(source_training_dataset,
                                         collate_fn=CollateSSDA(ssda_sampler=ssda_sampler),
                                         batch_size=config.pipeline.dataloader.train_batch_size,
                                         shuffle=True)

    validation_dataloader = get_dataloader(source_validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.train_batch_size*4,
                                           shuffle=False)

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    if config.adaptation.teacher_checkpoint:
        model = load_model(config.adaptation.teacher_checkpoint, model)
        print(f"--> Model loaded from checkpoint {config.adaptation.teacher_checkpoint}")
    else:
        raise ValueError("Pre-trained model needed for adaptation, check config.adaptation.teacher_checkpoint")

    pl_module = PLTTrainer(training_dataset=source_training_dataset,
                           validation_dataset=source_validation_dataset,
                           model=model,
                           criterion=config.adaptation.losses.source_criterion,
                           optimizer_name=config.pipeline.optimizer.name,
                           batch_size=config.pipeline.dataloader.train_batch_size,
                           val_batch_size=config.pipeline.dataloader.train_batch_size*4,
                           lr=config.pipeline.optimizer.lr,
                           num_classes=config.model.out_classes,
                           train_num_workers=config.pipeline.dataloader.num_workers,
                           val_num_workers=config.pipeline.dataloader.num_workers,
                           clear_cache_int=config.pipeline.lightning.clear_cache_int,
                           scheduler_name=config.pipeline.scheduler.name)

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    if config.pipeline.wandb.run_name is not None:
        run_name = run_time + '_' + config.pipeline.wandb.run_name
    else:
        run_name = run_time

    save_dir = os.path.join(config.pipeline.save_dir, run_name)

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               entity=config.pipeline.wandb.entity_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)

    loggers = [wandb_logger]

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'),
                                           save_on_train_epoch_end=True,
                                           every_n_epochs=1,
                                           save_top_k=-1)]

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator="ddp",
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=1.0,
                      num_sanity_val_steps=2,
                      resume_from_checkpoint=config.pipeline.lightning.resume_checkpoint,
                      callbacks=checkpoint_callback,
                      find_unused_parameters=False)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    train(config)
