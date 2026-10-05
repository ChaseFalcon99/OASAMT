# OASAMT

Occlusion-Aware Visual Object Tracking with Explicit Temporal State Modeling and Dual-Memory Mechanism

## Environment

This project is developed based on [SAM 2](https://github.com/facebookresearch/sam2) and uses the same environment configuration as the official SAM 2 implementation. Please follow the [SAM 2 installation instructions](https://github.com/facebookresearch/sam2#installation) to set up the required environment and dependencies.

## Getting Started

### Download the SAM 2.1 Checkpoint

```bash
cd checkpoints
./download_ckpts.sh
```

### Download the Pretrained TOC and TOP Models

Download the pretrained TOC and TOP models into the `tcn` directory.

| Model | Download |
|---|---|
| TOC Model | [toc_best_model.pt](https://github.com/ChaseFalcon99/OASAMT/releases/download/v0.1/toc_best_model.pt) |
| TOP Model | [top_best_model.pt](https://github.com/ChaseFalcon99/OASAMT/releases/download/v0.1/top_best_model.pt) |

### Run the Demo

```bash
cd demo
run demo.py
```

### Get the OccTrack Dataset

The OccTrack dataset is available on BaiduPan and can be downloaded from: 
| Dataset | Download |
|---|---|
| OccTrack Dataset | [BaiduPan](https://pan.baidu.com/s/1u_5ydYj9Dyb7G80oq42Ajw?pwd=m39s) |

Specifically, OASAMT_dataset is used to test OASAMT, TOC_dataset is used to train TOC, and TOP_dataset is used to train TOP.

### Train TOC and TOP

To train TOC and TOP, run:

```bash
cd toc_top/TOC
run train.py
```

```bash
cd toc_top/TOP
run train.py
```
