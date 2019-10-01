# Face-Adversarial-Attack

## Introduction
This is an easy approach for the competition "Facial Adversary Examples" in [TIANCHI](https://tianchi.aliyun.com/competition/entrance/231745/introduction?lang=en-us), which can get **3.5** in score based the evaluation criterion of 
the competition.


## Preparation
1. Download the dataset from [TIANCHI](https://tianchi.aliyun.com/competition/entrance/231745/introduction?lang=en-us). Suppose the directory is $DATA_DIR.

2. Download the pretrained Face-Recognition models from [Baidu](https://pan.baidu.com/s/1g0WNAqNQvqtB86JliYtQeQ) (Extraction code: sjqs).

3. Download the feature files from [Baidu](https://pan.baidu.com/s/1c5qsC5WdOPQFTfE8VMKzSg) (Extraction code: jf2z). Or you can use the script *attack/preprocess_eval.py* to generate these files.

4. Init attack mask directory:
        
    ```
    mkdir attack/masks
    ``` 

    Your directory tree should look like this:

    ```
    ${PROJECT_HOME}
    ├── attack
        ├── log
        ├── masks
        ├── state
        └── *.py
    ├── model
        └── downloaded models
    ├── result
        └── downloaded features
    ├── ...
    └── ...
    ```


## Dependencies
- python 3.6
- PyTorch 1.0.1
- CUDA 9.0
- CUDNN 7.1.2
- opencv 3.4.2 
- numpy 1.15
- scipy 1.2.0
### Note
- The code is developed using python 3.6 on Ubuntu 18.04. NVIDIA GPUs are needed.
- The code is tested using 1 NVIDIA 1080Ti GPU card. Other platforms or GPU cards are not fully tested.
- OpenCV is installed through anaconda, which is a little different with installed through pip.


## Usage
```(bash)
cd $PROJECT_HOME/attack

python attack.py \
    --root $DATA_DIR/securityAI_round1_images \
    --dev_path $DATA_DIR/securityAI_round1_dev.csv \
    --output_path $OUTPUT_PATH
```


## Acknowledgement
We develop our attack codes based wujiyang's [Face_Pytorch](https://github.com/wujiyang/Face_Pytorch).