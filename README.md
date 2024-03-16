# LUCF-Net
Code for paper "LUCF-Net: Lightweight U-shaped Cascade Fusion Network for Medical Image Segmentation". 

## 1. Environment

- Please prepare an environment with Ubuntu 20.04, with Python 3.9.16, PyTorch ≥ 2.0.0, and CUDA ≥ 11.7

## 2. Train/Test

- Train

```bash
python train.py  --root_path your DATA_DIR --max_epochs 600 --output_dir your OUT_DIR   --base_lr 0.05 --batch_size 16
```

- Test 

```bash
python test.py  --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --pretrained_pth model weights
```
## 3. model weights
https://drive.google.com/file/d/1damvHExKamIBf6_qh9L9K7APtrOnvYZA/view?usp=drive_link
## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [DAEFormert](https://github.com/xmindflow/DAEFormer)
```bibtex
@article{sun2024LUCF-Net,
  title={LUCF-Net: Lightweight U-shaped Cascade Fusion Network for Medical Image Segmentation},
  author={Sun, Songkai and She, Qingshan and Ma, Yuliang and Li, Rihui and Zhang， Yingchun},
  year={2024}
}
```