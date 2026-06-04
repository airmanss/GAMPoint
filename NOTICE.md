# Notice

This repository is built upon the Pointcept / Point Transformer V3 codebase.

The original Pointcept codebase is licensed under the MIT License. The original copyright and license notice are retained in the `LICENSE` file of this repository.

The GAMPoint-specific files provided in this repository include:

* `point_transformer_v3m1_base.py`: the modified Point Transformer V3 model implementation used for GAMPoint;
* `s3dis_config.py`: the configuration file for S3DIS experiments;
* `scannet_config.py`: the configuration file for ScanNet v2 experiments;
* `s3dis_preprocessing.py`: the preprocessing script for S3DIS;
* `scannet_preprocessing.py`: the preprocessing script for ScanNet v2.

These GAMPoint-specific files are provided to support reproducibility of the experiments reported in the manuscript.

This repository does not redistribute the raw or preprocessed files of ScanNet v2 or S3DIS. Users should obtain these datasets from the official dataset providers and follow the corresponding license and access requirements.

Please cite the original Pointcept / Point Transformer V3 work if you use this repository.
