<div align="center">

# Defect Detection based on Zero-Shot Instance Segmentation

</div>

## Description

What it does

## TODO
project 資料夾只寫定義，測試調參在 tests 寫，以 OO 去使用 project 內的 class
* [ ] model implementation
* [ ] training, testing/inference procedure (測試跟訓練調參統一寫在 tests 裡面，看要怎麼規劃流程才比較好做)
* [ ] dataset implementation
* [ ] ...

## How to run

First, install dependencies

```bash
# clone project
git clone https://github.com/guyleaf/Defect-Detection-ZSI
# or via SSH
git clone git@github.com:guyleaf/Defect-Detection-ZSI.git

# install project
cd Defect-Detection-ZSI
pip install -e .
pip install -r requirements.txt
```
## Dataset Implementation
### Dataset Keycap
Train image num : 218 <br>
Test image num : 46 <br>

### Input format
labels: 0, -1, >= 1
