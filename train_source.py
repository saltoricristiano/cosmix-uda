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
from utils.collation import CollateFN
from utils.pipelines import PLTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar2semantickitti.yaml",
                    type=str,
                    help="Path to config file")


def train(config):

    def get_dataloader(dataset, batch_size, collate_fn=CollateFN(), shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        mapping_path = config.dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None

    training_dataset, validation_dataset, target_dataset = get_dataset(dataset_name=config.dataset.name,
                                                                       dataset_path=config.dataset.dataset_path,
                                                                       target_name=config.dataset.target,
                                                                       voxel_size=config.dataset.voxel_size,
                                                                       augment_data=config.dataset.augment_data,
                                                                       version=config.dataset.version,
                                                                       sub_num=config.dataset.num_pts,
                                                                       num_classes=config.model.out_classes,
                                                                       ignore_label=config.dataset.ignore_label,
                                                                       mapping_path=mapping_path)

    collation = CollateFN()
    training_dataloader = get_dataloader(training_dataset,
                                         collate_fn=collation,
                                         batch_size=config.pipeline.dataloader.batch_size,
                                         shuffle=True)

    validation_dataloader = get_dataloader(validation_dataset,
                                           collate_fn=collation,
                                           batch_size=config.pipeline.dataloader.batch_size*4,
                                           shuffle=False)

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    pl_module = PLTTrainer(training_dataset=training_dataset,
                           validation_dataset=validation_dataset,
                           model=model,
                           criterion=config.pipeline.loss,
                           optimizer_name=config.pipeline.optimizer.name,
                           batch_size=config.pipeline.dataloader.batch_size,
                           val_batch_size=config.pipeline.dataloader.batch_size*4,
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

    checkpoint_callback = [ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1)]

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
                      callbacks=checkpoint_callback)

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
