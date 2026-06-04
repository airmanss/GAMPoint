# GAMPoint

This repository provides the implementation and reproducibility materials for **GAMPoint**, a geometry-augmented transformer framework for 3D point-cloud semantic segmentation.

GAMPoint is implemented based on the **Pointcept / Point Transformer V3** codebase. This repository provides the base Pointcept code package, the GAMPoint model implementation, dataset configuration files, and preprocessing scripts used in the manuscript.

## Repository structure

```text
GAMPoint/
├── README.md
├── LICENSE
├── NOTICE.md
├── Pointcept-main.zip
├── point_transformer_v3m1_base.py
├── s3dis_config.py
├── scannet_config.py
├── s3dis_preprocessing.py
└── scannet_preprocessing.py
```

## Files

### Base code

```text
Pointcept-main.zip
```

This file contains the Pointcept codebase used as the base framework. Please unzip it before running the code.

### GAMPoint model

```text
point_transformer_v3m1_base.py
```

This file contains the modified Point Transformer V3 model implementation used for GAMPoint in the manuscript.

### Configuration files

```text
s3dis_config.py
scannet_config.py
```

These files provide the experimental configurations for the two benchmark datasets used in the manuscript:

* `s3dis_config.py`: configuration for S3DIS experiments;
* `scannet_config.py`: configuration for ScanNet v2 experiments.

### Preprocessing scripts

```text
s3dis_preprocessing.py
scannet_preprocessing.py
```

These files provide the preprocessing scripts used for preparing the S3DIS and ScanNet v2 datasets following the Pointcept data preparation pipeline.

## Environment setup

This repository follows the environment requirements of Pointcept / Point Transformer V3.

Please first unzip the base code:

```bash
unzip Pointcept-main.zip
```

Then configure the environment according to the official Pointcept instructions:

```text
https://github.com/Pointcept/Pointcept
```

After the Pointcept environment is configured, place the GAMPoint model file and configuration files into the corresponding Pointcept project directories before training or evaluation.

## Data preparation

This repository does **not** redistribute the raw or preprocessed files of ScanNet v2 or S3DIS.

ScanNet v2 and S3DIS are third-party benchmark datasets. Users should obtain them from their official dataset providers and follow the corresponding license and access requirements.

In this work, we followed the Pointcept data preparation pipeline for ScanNet v2 and S3DIS. Please refer to the official Pointcept data preparation instructions:

```text
https://github.com/Pointcept/Pointcept#data-preparation
```

After obtaining the datasets from the official sources, users can use the following preprocessing scripts provided in this repository:

```text
s3dis_preprocessing.py
scannet_preprocessing.py
```

Please modify the dataset paths in the preprocessing scripts and configuration files according to your local environment.

## Training and evaluation

The training and evaluation procedures follow the standard Pointcept workflow.

Example commands:

```bash
# Train on S3DIS
python tools/train.py --config-file s3dis_config.py

# Train on ScanNet v2
python tools/train.py --config-file scannet_config.py
```

Please adjust the dataset paths, output paths, GPU settings, and other environment-specific parameters in the configuration files before running the experiments.

## Data availability

This repository provides the code and configuration files required to reproduce the GAMPoint experiments reported in the manuscript.

The ScanNet v2 and S3DIS datasets are not redistributed in this repository. Users should obtain the datasets from their official sources and prepare them following the Pointcept data preparation pipeline.

The experimental result files supporting the manuscript findings, such as main results, ablation results, repeated-run results, per-class metrics, variance analysis, and statistical tests, should be provided separately as the minimal data underlying the manuscript findings.

## License and acknowledgement

This repository is built upon the Pointcept codebase, which is released under the MIT License. The original Pointcept license and copyright notice are retained in the `LICENSE` file.

The GAMPoint-specific model implementation, configuration files, and preprocessing scripts are provided for academic research and reproducibility.

Please cite the original Pointcept / Point Transformer V3 work if you use this repository.
