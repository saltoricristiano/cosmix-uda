# **CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation [ECCV2022]**

The official implementation of our work "CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation".



![video](https://user-images.githubusercontent.com/56728964/177744041-9a486a32-e2d5-4049-a524-692830eae2be.mp4)



## Introduction

Several Unsupervised Domain Adaptation (UDA) methods for point cloud data have been recently proposed to improve model generalization for different sensors and environments.
Meanwhile, researchers working on UDA problems in the image domain have shown that sample mixing can mitigate domain shift.
We propose a new approach of sample mixing for point cloud UDA, namely Compositional Semantic Mix (CoSMix), the first UDA approach for point cloud segmentation based on sample mixing.
CoSMix consists of a two-branch symmetric network that can process synthetic labelled data (source) and real-world unlabelled point clouds (target) concurrently.
Each branch operates on one domain by mixing selected pieces of data from the other one, and by using the semantic information derived from source labels and target pseudo-labels.

For more information follow the [PAPER]() link (:fire: COOMING SOON :fire:)!

Authors: [Cristiano Saltori](https://scholar.google.com/citations?user=PID7Z4oAAAAJ&hl),
         [Fabio Galasso](https://scholar.google.com/citations?user=2gSuGBEAAAAJ&hl),
         [Giuseppe Fiameni](https://scholar.google.com/citations?user=Se2mLvIAAAAJ&hl),
         [Nicu Sebe](https://scholar.google.it/citations?user=tNtjSewAAAAJ&hl),
         [Elisa Ricci](https://scholar.google.ca/citations?user=xf1T870AAAAJ&hl),
         [Fabio Poiesi](https://scholar.google.co.uk/citations?user=BQ7li6AAAAAJ&hl)

![teaser](assets/mix_teaser_complex.jpg)

## News :bell:
- 7/2022: CoSMix code has been **RELEASED**!
- 7/2022: CoSMix is accepted to ECCV 2022!:fire: Our work is the first using compositional mix between domains to allow adaptation in LiDAR segmentation!

## Installation
The code has been tested with Docker (see Docker container below) with Python 3.8, CUDA 10.2/11.1, pytorch 1.8.0 and pytorch-lighting 1.4.1.
Any other version may requireq to update the code for compatibility.

### Pip/Venv/Conda
In your virtual environment follow [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine).
This will install all the base packages.

Additionally, you need to install:
- [open3d 0.13.0](http://www.open3d.org)
- [pytorch-lighting 1.4.1](https://www.pytorchlightning.ai)
- [wandb](https://docs.wandb.ai/quickstart)
- tqdm
- pickle


### Docker container
If you want to use Docker you can find a ready-to-use container at ```crissalto/online-adaptation-mink:1.3```, just be sure to have installed drivers compatible with CUDA 11.1.




## Data preparation

### SynLiDAR
Download SynLiDAR dataset from [here](https://github.com/xiaoaoran/SynLiDAR), then prepare data folders as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    └──sequences/
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 12/
```

### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

### SemanticPOSS
To download SemanticPOSS follow the instructions [here](http://www.poss.pku.edu.cn/semanticposs.html). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── tag
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── instances.txt
            └── 06/
```


After you downloaded the datasets you need, create soft-links in the ```data``` directory
```
cd cosmix-uda
mkdir data
ln -s PATH/TO/SEMANTICKITTI SemanticKITTI
# do the same for the other datasets
```

## Source training

We use SynLiDAR as source synthetic dataset. 
The first stage of CoSMix consists of a warm-up stage in which the teacher is pretrained on the source dataset.
To warm-up the segmentation model on SynLiDAR2SemanticKITTI run

```
python train_source.py --config_file configs/source/synlidar2semantickitti.yaml
```


while to warm-up the segmentation model on SynLiDAR2SemanticPOSS run

```
python train_source.py --config_file configs/source/synlidar2semanticposs.yaml
```

**NB:** we provide source pretrained models, so you can skip this step and move directly on adaptation! :rocket:

## Pretrained models :rocket:

You can download the pretrained models on both SynLiDAR2SemanticKITTI and SynLiDAR2SemanticPOSS form [here](https://drive.google.com/file/d/1cwRaIobmU0-DKDaic6Y7UX03O7MAFBV5/view?usp=sharing) and decompress them in ```cosmix-uda/pretrained_models/```.


## Target adaptation

To adapt with CoSMix on SynLiDAR2SemanticKITTI run

```
python adapt_cosmix.py --config_file configs/adaptation/synlidar2semantickitti_cosmix.yaml
```

while to adapt with CoSMix on SynLiDAR2SemanticPOSS run

```
python adapt_cosmix.py --config_file configs/adaptation/synlidar2semanticposs_cosmix.yaml
```

## Evaluation

To evaluate pretrained models after warm-up
```
python eval.py --config_file configs/config-file-of-the-experiment.yaml --resume_path PATH-TO-EXPERIMENT
```

with ```--eval_source``` for running evaluation on source data and ```--eval_target``` on target data.
This will iterate over all the checkpoints and run evaluation of all the checkpoints in ```PATH-TO-EXPERIMENT/checkpoints/```.

Similarly, after adaptation use
```
python eval.py --config_file configs/config-file-of-the-experiment.yaml --resume_path PATH-TO-EXPERIMENT --is_student
```

Where ```--is_student``` specifies that the model to be evaluated is a student model.

You can save predictions for future visualizations by adding ```--save_predictions```.

## References
References will be uploaded SOON !:rocket:

## Acknowledgments
The work was partially supported by OSRAM GmbH,  by the Italian Ministry of Education, Universities and Research (MIUR) ”Dipartimenti di Eccellenza 2018-2022”, by the SHIELD project, funded by the European Union’s Joint Programming Initiative – Cultural Heritage, Conservation, Protection and Use joint call and, it was carried out in the Vision and Learning joint laboratory of FBK and UNITN.


## Thanks

We thank the opensource project [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
