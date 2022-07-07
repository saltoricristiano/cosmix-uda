# **CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation [ECCV2022]**

The official implementation of our work "CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation".

![video](assets/cosmix.mov)

## Introduction

Several Unsupervised Domain Adaptation (UDA) methods for point cloud data have been recently proposed to improve model generalization for different sensors and environments.
Meanwhile, researchers working on UDA problems in the image domain have shown that sample mixing can mitigate domain shift.
We propose a new approach of sample mixing for point cloud UDA, namely Compositional Semantic Mix (CoSMix), the first UDA approach for point cloud segmentation based on sample mixing.
CoSMix consists of a two-branch symmetric network that can process synthetic labelled data (source) and real-world unlabelled point clouds (target) concurrently.
Each branch operates on one domain by mixing selected pieces of data from the other one, and by using the semantic information derived from source labels and target pseudo-labels.

For more information follow the [PAPER]() link (:fire: COOMING SOON :fire:)!

![teaser](assets/mix_teaser_complex.jpg)

## News :bell:
- 7/2022: CoSMix code is **OUT**!
- 7/2022: CoSMix is accepted to ECCV 2022!:fire:
- 7/2022: CoSMix repository has been created! Our work is the first using compositional mix between domains to allow adaptation in LiDAR segmentation!

## Installation


### Pip/Venv/Conda



### Docker



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
To download SemanticPOSS follow the instructions [here](). Then, prepare the paths as follows:
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


After you downloaded the datasets you need, create soft-links in the ```data``` directory
```
cd cosmix-uda
mkdir data
ln -s PATH/TO/SEMANTICKITTI SemanticKITTI
# do the same for the other datasets
```

## Source training

We use SynLiDAR as source synthetic dataset. To train the source model on SynLIDAR use
```
python 
```

**NB:** we provide pretrained models in ```pretrained_models```, so you can skip this time consuming step!:rocket:

## Target adaptation

Adapt to the target domain with

```
python 
```

## Acknowledgments
The work was partially supported by OSRAM GmbH,  by the Italian Ministry of Education, Universities and Research (MIUR) ”Dipartimenti di Eccellenza 2018-2022”, by the SHIELD project, funded by the European Union’s Joint Programming Initiative – Cultural Heritage, Conservation, Protection and Use joint call and, it was carried out in the Vision and Learning joint laboratory of FBK and UNITN.


## Thanks