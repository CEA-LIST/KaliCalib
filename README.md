
# KaliCalib: A Framework for Basketball Court Registration

KaliCalib is a basketball court registration framework developped to participate to the [ACM MMSports 2022 camera calibration challenge](https://eval.ai/web/challenges/challenge-page/1687).
The method is described in our paper "KaliCalib: A Framework for Basketball Court Registration" accepted as a challenge paper to MMSports 2022. You can find an ArXiv version [here](https://arxiv.org/abs/2209.07795).
This source code is available under the CeCILL 2.1 license.

## Installation

Create a virtual environment:
```
virtualenv venv
source venv/bin/activate
```

Install the dependancies:
```
pip install -r requirements.txt
```

## Dataset

Please follow the [instructions](https://github.com/DeepSportRadar/camera-calibration-challenge#download-and-prepare-the-dataset) available on the challenge repository to download and prepare the dataset.

## Usage

To train the model only with the challenge train dataset, you can use the `train.sh` script.
Evaluation on the test dataset can be run with the `eval.sh` script. By default, this script loads the provided `model_test.pth` model but this can be modified in the `eval_test.yml` config file.

The challenge organizers allowed to use the complete dataset (train, test and validation data) to train a model for an evaluation with the challenge data.
You can achieve this with the `train_full_dataset.sh` script. The evaluation on the challenge data can be run with the `eval_challenge.sh` script. By default, this script loads the provided `model_challenge.pth` model but this can be modified in the `eval_challenge.yml` config file.

## Results

Test model MSE on the test dataset: **107.78 mm**.

Challenge model MSE on the challenge dataset: **73.16 mm**.
