<h1 align="center"> UAV-206 </h1>


全球校园人工智能算法精英大赛-基于无人机的人体行为识别

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, h5py, sklearn, matplotlib, thop
- Run `pip install -e torchpack`
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 2 datasets to download:


1. Download dataset here: https://drive.google.com/file/d/1H-U_cDg_S-LQRWtwJ_imDXfYDT__4OeY/view?usp=sharing
2. unzip the datasets zip `data.zip` to `/data/`

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - train_joint.npy
  - train_label.npy
  - val_joint.npy
  - val_label.npy
  - test_joint.npy
  - test_label.npy
  - gen_empty_test_label.py  #be used to generate `test_label.npy`
```

# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training DeGCN with joint modality
python main.py --config config/degcn/train_j.yaml --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

## Acknowledgements

This repo is based on [DEGCN](https://github.com/WoominM/DeGCN_pytorch).

Thanks to the original authors for their work!
