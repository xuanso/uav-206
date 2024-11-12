<h1 align="center"> UAV-206 </h1>


全球校园人工智能算法精英大赛-基于无人机的人体行为识别

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX, h5py, sklearn, matplotlib, thop
- Run `pip install -e torchpack`
- Run `pip install -e torchlight` 

```
#install apex

pip uninstall setuptools

pip install setuptools==60.2.0

pip install packaging

rm -R apex

git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

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

python main.py --config config/degcn/train_j.yaml
python main.py --config config/degcn/train_b.yaml
python main.py --config config/degcn/train_jbf.yaml

python main_mixf.py --config config/skeleton-mixformer/train_j.yaml
python main_mixf.py --config config/skeleton-mixformer/train_b.yaml

python main_stt.py --config config/sttformer/train_j.yaml
python main_stt.py --config config/sttformer/train_b.yaml
python main_stt.py --config config/sttformer/train_jm.yaml

python main_tdgcn.py --config config/tdgcn/train_j.yaml
python main_tdgcn.py --config config/tdgcn/train_b.yaml

cd infogcn
python main.py --config config/train_j.yaml

```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```

python main.py --config config/degcn/test_*.yaml --phase test --save-score True --weights work_dir/degcn/xxx.pt

python main_mixf.py --config config/skeleton-mixformer/test_*.yaml --phase test --save-score True --weights work_dir/skeleton-mixformer/xxx.pt

python main_stt.py --config config/sttformer/test_*.yaml --phase test --save-score True --weights work_dir/sttformer/xxx.pt

python main_tdgcn.py --config config/tdgcn/test_*.yaml --phase test --save-score True --weights work_dir/tdgcn/xxx.pt

cd infogcn
python main.py --config config/test_j.yaml --phase test --save-score True --weights ../work_dir/infogcn/xxx.pt

```

