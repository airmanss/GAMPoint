# GAMPoint

This repository provides the implementation and reproducibility materials for **GAMPoint**, a geometry-augmented transformer framework for 3D point-cloud semantic segmentation.

GAMPoint is implemented based on the **Pointcept / Point Transformer V3** codebase. This repository provides the base Pointcept code package, the GAMPoint model implementation, dataset configuration files, preprocessing scripts, and the minimal result data used to support the manuscript.

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
├── scannet_preprocessing.py
└── paper_results/
    ├── README.md
    ├── logs_excerpt/
    │   ├── s3dis_input_modality_and_repeated_runs_excerpt_corrected.txt
    │   ├── s3dis_window_size_sensitivity_excerpt.txt
    │   └── scannet_table4_results_excerpt.txt
    ├── s3dis_input_modality_results_from_logs.csv
    ├── s3dis_paper_selected_results_from_logs.csv
    ├── s3dis_repeated_runs_from_logs.csv
    ├── s3dis_repeated_runs_summary.csv
    ├── s3dis_window_size_sensitivity_from_logs.csv
    ├── s3dis_window_size_sensitivity_summary.csv
    ├── scannet_main_results_from_table.csv
    └── scannet_per_class_iou_from_table.csv
```

## Files

### Base code

```text
Pointcept-main.zip
```

This file contains the Pointcept codebase used as the base framework. Please unzip this file before using the code.

### GAMPoint model implementation

```text
point_transformer_v3m1_base.py
```

This file contains the modified Point Transformer V3 model implementation used for GAMPoint in the manuscript.

### Configuration files

```text
s3dis_config.py
scannet_config.py
```

These files provide the experimental configurations used for the two benchmark datasets in the manuscript:

* `s3dis_config.py`: configuration for S3DIS experiments;
* `scannet_config.py`: configuration for ScanNet v2 experiments.

### Preprocessing scripts

```text
s3dis_preprocessing.py
scannet_preprocessing.py
```

These files provide the preprocessing scripts used for S3DIS and ScanNet v2 in this work.

## Environment setup

This repository follows the environment requirements of Pointcept / Point Transformer V3.

Please configure the environment according to the official Pointcept / Point Transformer V3 instructions. The environment requirements, dependencies, and official running scripts should be checked from the original Pointcept documentation.

The base Pointcept package is provided as:

```text
Pointcept-main.zip
```

After extracting the base code, place the GAMPoint model file, configuration files, and preprocessing scripts into the corresponding Pointcept project locations before running experiments.

## Data preparation

This repository does **not** redistribute the raw or preprocessed files of ScanNet v2 or S3DIS.

ScanNet v2 and S3DIS are third-party benchmark datasets. Users should obtain these datasets from their official dataset providers and follow the corresponding license and access requirements.

In this work, we followed the Pointcept data preparation pipeline for ScanNet v2 and S3DIS:

```text
https://github.com/Pointcept/Pointcept#data-preparation
```

After obtaining the datasets from the official sources, users may refer to the Pointcept data preparation instructions and the preprocessing scripts provided in this repository:

```text
s3dis_preprocessing.py
scannet_preprocessing.py
```

Please modify dataset paths in the preprocessing scripts and configuration files according to the local environment.

## Training and evaluation

Training and evaluation follow the official Pointcept workflow.

This repository does not redefine the official Pointcept running commands. Users should refer to the original Pointcept / Point Transformer V3 documentation for the exact training and evaluation commands, and use the GAMPoint model file and configuration files provided in this repository.

The configuration files used for the experiments in the manuscript are:

```text
s3dis_config.py
scannet_config.py
```

## Minimal result data

The minimal result data underlying the manuscript findings are provided in:

```text
paper_results/
```

This folder contains structured CSV files and result excerpts used to support the reported experimental results.

### S3DIS result files

```text
paper_results/s3dis_input_modality_results_from_logs.csv
paper_results/s3dis_paper_selected_results_from_logs.csv
paper_results/s3dis_repeated_runs_from_logs.csv
paper_results/s3dis_repeated_runs_summary.csv
paper_results/s3dis_window_size_sensitivity_from_logs.csv
paper_results/s3dis_window_size_sensitivity_summary.csv
```

These files include S3DIS input-attribute results, paper-selected results, repeated-run results, summary statistics, and window-size sensitivity results.

### ScanNet result files

```text
paper_results/scannet_main_results_from_table.csv
paper_results/scannet_per_class_iou_from_table.csv
```

These files include the ScanNet total mIoU values and per-class IoU values transcribed from the available ScanNet result table.

### Result excerpts

```text
paper_results/logs_excerpt/
```

This folder contains the text excerpts corresponding to the reported results. Each CSV file includes a `source_log` or `source_file` field to indicate the source excerpt used to obtain the values.

The full raw ScanNet v2 and S3DIS datasets are not included in this repository.

## Notes on data availability

The raw and preprocessed ScanNet v2 and S3DIS dataset files are not redistributed here. Users should obtain the datasets from their official providers and prepare them according to the Pointcept data preparation pipeline.

The files in `paper_results/` provide the minimal result data used to support the experimental findings reported in the manuscript.

## License and acknowledgement

This repository is built upon the Pointcept codebase, which is released under the MIT License. The original Pointcept license and copyright notice are retained in the `LICENSE` file.

The GAMPoint-specific model implementation, configuration files, preprocessing scripts, and result summaries are provided for academic research and reproducibility.

Please cite the original Pointcept / Point Transformer V3 work if you use this repository.
