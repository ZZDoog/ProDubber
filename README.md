# Prosody-Enhanced Acoustic Pre-training and Acoustic-Disentangled Prosody Adapting for Movie Dubbing

[CVPR2025] Official implementation of paper "Prosody-Enhanced Acoustic Pre-training and Acoustic-Disentangled \\ Prosody Adapting for Movie Dubbing"

## 🌼 Environment

Our python version is ```3.8.18``` and cuda version ```11.8```. It's possible to have another compatible version. 
Both training and inference are implemented with PyTorch on a
GeForce RTX 4090 GPU.

```
conda create -n dubbing python=3.8.18
conda activate dubbing
pip install -r requirements.txt
```

## 🔧 Training

#### For First Stage (Acoustic Pre-training)
```
python train_first.py -p Configs/config_v2c_stage1.yml  # V2C-Animation benchmark
python train_first.py -p Configs/config_grid_stage1.yml  # GRID benchmark
```
#### For Second Stage (Prosody Adapting)
```
python train_second.py -p Configs/config_v2c.yml  # V2C-Animation benchmark
python train_second_grid.py -p Configs/config_grid.yml  # GRID benchmark
```

## 💡 Checkpoints

We provide the first stage and second stage pre-trained checkpoints on V2C-Animation and GRID benchmarks as follows, respectively:

#### First stage (For secnod stage only, can not directly generate wavform)
- V2C-Animation benchmark: [Baidu Drive](https://pan.baidu.com/s/1ZUAjOu4jTkx0znVMBngVKA) (b5wy), [Google Drive](https://drive.google.com/file/d/1HF1Bh44oO8w2EYOfX0H6ZhUHVFhMRJOG/view?usp=drive_link).

- GRID benchmark: [Baidu Drive](https://pan.baidu.com/s/1evzJYGRhqoyLg7f9kTUyiw) (wj25), [Google Drive](https://drive.google.com/file/d/1kMIxHi3_ISFsChCsbnJOcHqpe7lk9_XP/view?usp=drive_link)

#### Second stage (Can used to directly generate waveform)
- V2C-Animation benchmark: [Baidu Drive](https://pan.baidu.com/s/1BfPELKc6BVcX9Vz4KnbToA) (3k4h), [Google Drive](https://drive.google.com/file/d/1VucTEmMVNpIJLDtvJbX7lv4VqubVC7we/view?usp=drive_link).

- GRID benchmark: [Baidu Drive](https://pan.baidu.com/s/1Nt9G7Xp9aEnlNTC9KoJ0fg) (23vd), [Google Drive](https://drive.google.com/file/d/1pk9gcGUcM5OnibcxktUwYM-m73QxPtUV/view?usp=drive_link)

## ✍ Inference

### For V2C-Animation Benchmark
There is three generation settings in V2C-Animation benchmark:

```
python inference_v2c.py -n 'YOUR_EXP_NAME' --epoch 'YOUR_EPOCH' --setting 1
```
```
python inference_v2c.py -n 'YOUR_EXP_NAME' --epoch 'YOUR_EPOCH' --setting 2
```
```
python inference_v2c.py -n 'YOUR_EXP_NAME' --epoch 'YOUR_EPOCH' --setting 3
```

### For GRID benchmark
There is two generation settings in GRID benchmark:
```
python inference_grid.py -n 'YOUR_EXP_NAME' --epoch 'YOUR_EPOCH' --setting 1
```
```
python inference_grid.py -n 'YOUR_EXP_NAME' --epoch 'YOUR_EPOCH' --setting 2
```

## 📊 Dataset

- GRID ([BaiduDrive](https://pan.baidu.com/s/1E4cPbDvw_Zfk3_F8qoM7JA) (code: GRID) /  [GoogleDrive](https://drive.google.com/drive/folders/1_z51hy6H3K4kyHy-MXtMfo2Py6edpscE?usp=drive_link))
- V2C-Animation dataset (chenqi-Denoise2) ([BaiduDrive]( https://pan.baidu.com/s/12hEFbXwpv4JscG3tUffjbA) (code: k9mb) / [GoogleDrive](https://drive.google.com/drive/folders/11WhRulJd23XzeuWmUVay5carpudGq3ig?usp=drive_link))


## 🙏 Acknowledgments
We would like to thank the authors of previous related projects for generously sharing their code and insights: [StyleTTS](https://github.com/yl4579/StyleTTS), [StyleTTS2](https://github.com/yl4579/StyleTTS2), [StyleDubber](https://github.com/GalaxyCong/StyleDubber), [PL-BERT](https://github.com/yl4579/PL-BERT), and [HiFi-GAN](https://github.com/jik876/hifi-gan).


## 🤝 Ciation
If you find our work useful, please consider citing:
```
@misc{zhang2025produbber,
      title={Prosody-Enhanced Acoustic Pre-training and Acoustic-Disentangled Prosody Adapting for Movie Dubbing}, 
      author={Zhedong Zhang and Liang Li and Chenggang Yan and Chunshan Liu and Anton van den Hengel and Yuankai Qi},
      year={2025},
      eprint={2503.12042},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2503.12042}, 
}
```