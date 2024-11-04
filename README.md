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


1. Download dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
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
# Example: training DeGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/degcn --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training DeGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train-feeder-args bone=True --test-feeder-args bone=True --work-dir work_dir/ntu120/csub/degcn_bone --device 0
```

- To train model the JBF stream, setting `model` arguments in the config file `default.yaml` or in the command line.

```
# Example: training DeGCN with the JBF stream on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.jbf.Model --work-dir work_dir/ntu120/csub/degcn_bone --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of DeGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/xsub/degcn --bone-dir work_dir/ntu120/xsub/degcn_bone --joint-motion-dir work_dir/ntu120/xsub/degcn_motion
```

<!-- ### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject .
- Put files to <work_dir> and run **Testing** command to produce the final result. -->


## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!


# Citation

Please cite this work if you find it useful:.

      @inproceedings{,
        title={DeGCN: Deformable Graph Convolutional Networks for Skeleton-Based Action Recognition},
        author={Woomin Myung, Nan Su, Jing-Hao Xue, Guijin Wang},
        journal={IEEE transactions on image processing (TIP)}
        year={2024}
      }
