# Paper Results

This folder provides the minimal result data underlying the S3DIS experimental findings reported in the manuscript.

The raw and preprocessed ScanNet v2 and S3DIS dataset files are not redistributed in this repository. These datasets should be obtained from their official providers under the corresponding license and access requirements.

## Files

- `s3dis_repeated_runs_from_logs.csv`: raw five-run results for GAMPoint and BaseOpt with RGB+XYZ+Normal input.
- `s3dis_repeated_runs_summary.csv`: mean and sample standard deviation computed from `s3dis_repeated_runs_from_logs.csv`.
- `s3dis_input_modality_results_from_logs.csv`: S3DIS input-modality results using `XYZ`, `RGB+XYZ`, and `RGB+XYZ+Normal`.
- `s3dis_paper_selected_results_from_logs.csv`: rows marked as paper results in the source excerpt.
- `s3dis_window_size_sensitivity_from_logs.csv`: raw window-size sensitivity results.
- `s3dis_window_size_sensitivity_summary.csv`: mean and sample standard deviation for the window-size sensitivity experiment.
- `logs_excerpt/`: text excerpts corresponding to the result logs.

Each row in the CSV files includes a `source_log` or `source_file` field indicating the source file from which the result was extracted.

## Notes

For `RGB+XYZ+Normal`, the source excerpt marks one of the five runs as the paper result. Therefore, this package provides both:

1. the paper-selected result row; and
2. the five-run mean and sample standard deviation for transparency.

For `RGB+XYZ` and `XYZ`, only single-run results are available in the source excerpt.

## Blank cells

No CSV cells are intentionally left blank. Fields that are not applicable are marked as `NA`.

## ScanNet table results

The following files provide the ScanNet results transcribed from the user-provided Table 4 screenshot:

- `scannet_main_results_from_table.csv`: total ScanNet mIoU values for the table columns `BaseOpt`, `GAMPoint w/o EnhancedRPE`, and `GAMPoint`.
- `scannet_per_class_iou_from_table.csv`: per-class ScanNet mIoU values for the same table columns.
- `logs_excerpt/scannet_table4_results_excerpt.txt`: text excerpt of the transcribed ScanNet table.
- `logs_excerpt/scannet_table4_screenshot.png`: screenshot source provided for the ScanNet table.

No full ScanNet training logs are included in this folder. The available source for these ScanNet values is the table screenshot/excerpt.
