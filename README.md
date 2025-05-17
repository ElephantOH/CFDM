<hr>
<h1 align="center">
  Cascaded Fractal Diffusion Model <br>
  <sub>Cascaded Fractal Diffusion Model based on Master-Slave Structure for Fast Cross-Modality Generation</sub>
</h1>
<hr>

<hr>

[//]: # (<h3 align="center">[<a href="https://arxiv.org/">arXiv</a>]</h3>)

Official PyTorch implementation of **CFDM**. Experiments demonstrate that our method performs effectively across two medical datasets and three thermal infrared visible light facial datasets.

<p align="center">
  <img src="figures/frame.png" alt="frame" style="width: 1200px; height: auto;">
</p>

## 🐹 Installation

This repository has been developed and tested with `CUDA 11.7` and `Python 3.8`. Below commands create a conda environment with required packages. Make sure conda is installed.

```
conda env create --file requirements.yaml
conda activate cfdm
```

## 🐼 Prepare dataset
The default data set class GetDataset requires a specific folder structure for organizing the data set.
Modalities (such as `T1, T2, etc.`) should be stored in separate folders, while splits `(train, test, and optionally val)` should be arranged as subfolders containing `2D` images named `slice_0.png or .npy, slice_1.png or .npy`, and so on.
To utilize your custom data set class, implement your version in `dataset.py` by inheriting from the `CFDMDataset` class.

```
<datasets>/
├── <modality_a>/
│   ├── train/
│   │   ├── T1
│   │   │   ├── slice_0.png or .npy
│   │   │   ├── slice_1.png or .npy
│   │   │   └── ...
│   │   └── T2
│   │   │   ├── slice_0.png or .npy
│   │   │   ├── slice_1.png or .npy
│   │   │   └── ...
│   ├── test/
│   │   ├── T1
│   │   └── ...
│   └── val/ (The file does not exist by default)
│       ├── T1
│       └── ...
├── <modality_b>/
│   ├── train/
│   ├── test/
│   └── val/ (The file does not exist by default)
├── ...
  
```

## 🙉 Training Model

Execute the following command to start or resume training.
Model checkpoints are stored in the `/checkpoints/$LOG` directory.
The script supports both `single-GPU` and `multi-GPU` training, with `single-GPU` mode being the default setting.

The example training code is as follows: 
```
python train_CFDM.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 2 \
  --max_epoch 120 \
  --lr 1.5e-4 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_CFDM_logs
```

### Argument descriptions

| Argument                  | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| `--input_channels`        | Dimension of images.                                                                                  |
| `--source` and `--target` | Source Modality and Target Modality, e.g. 'T1', 'T2'. Should match the folder name for that modality. |
| `--batch_size`            | Train set batch size.                                                                                 |
| `--lr`                    | Learning rate.                                                                                        |
| `--max_epoch`             | Number of training epochs (default: 120).                                                             |
| `--input_path`            | Data set directory.                                                                                   |
| `--checkpoint_path`       | Model checkpoint path to resume training.                                                             |


## 🐧 Training Noise Gate

Run the following command to start tuning.
The predicted images are saved under `/checkpoints/$LOG/generated_samples` directory.
By default, the script runs on a `single GPU`. 

```
python train_NoiseGate.py \
  --input_channels 1 \
  --source T1 \
  --target T2 \
  --batch_size 2 \
  --tuning_dataset_num 200 \
  --which_epoch 120 \
  --gpu_chose 0 \
  --input_path ./datasets/BrainTs20 \
  --checkpoint_path ./checkpoints/brats_1to2_CFDM_logs
```

## 🐣 Testing

Run the following command to start testing.
The predicted images are saved under `/checkpoints/$LOG/generated_samples` directory.
By default, the script runs on a `single GPU`. 

```
python test_CFDM.py \
        --input_channels 1 \
        --source T1 \
        --target T2 \
        --batch_size 2 \
        --which_epoch 120 \
        --gpu_chose 0 \
        --input_path ./datasets/BrainTs20 \
        --checkpoint_path ./checkpoints/brats_1to2_CFDM_logs
```

### Argument descriptions

Some arguments are common to both training and testing and are not listed here. For details on those arguments, please refer to the training section.

| Argument        | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `--batch_size`  | Test set batch size.                                                  |
| `--which_epoch` | Model checkpoint path.                                                |

## 🐸 Checkpoint

Refer to the testing section above to perform inference with the checkpoints. PSNR (dB), SSIM (%) and MAE are listed as mean ± std across the test set.

The paper is currently undergoing blind review. If you need weight open source or code open source, please contact us `elephantoh@qq.com`.

| Dataset | Task      | PSNR | SSIM | MAE | Checkpoint                  |
|---------|-----------|------|------|-----|-----------------------------|
| LLVIP   | VIS.→THE. | -    | -    | -   | [Link](https://github.com/) |
| TFW     | VIS.→THE. | -    | -    | -   | -                           |
| Tufts   | VIS.→THE. | -    | -    | -   | -                           |
| BrainTs | T1→T2     | -    | -    | -   | -                           |
| OASIS3  | T1→T2     | -    | -    | -   | -                           |



## 🦊 Code

The code for the `test` is open, and the code for the `train` will be made public shortly.

## 🐭 Citation

You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```

```

<hr>