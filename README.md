## This repository contains code for the Performance of the Non-Linear Kalman Filter on Person Re-Identification and Tracking

##This is the code for the Extended Kalman Filter

## Dependencies

The code is compatible with Python 2.7 to 3.6. The following dependencies are
needed to run the tracker:

* Python Version: 3.6
*OpenCV Version: 4.5.2
*Numpy Version: 1.19.5
*Scikit-Learn Version: 0.22.2
*iMutils Version : 0.5.4
*Pillow(PIL) Version : 8.2.0
*Tensorflow Version: 1.14.0


## 1: Downloading the Validation Dataset
Download the MOT16 dataset from this website: https://motchallenge.net/data/MOT16/ and place it in the root directory

## 2: Running the tracker
Use the following to run the tracker on any of the MOT16 sequence: 
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True

This is one I used : python deep_sort_app.py --sequence_dir=./MOT16/test/MOT16-06 --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy --min_confidence=0.3 --nn_budget=100 --display=True

##3 Generating Detections and saving it numpy array file(.npy), use the below code:
python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=./MOT16/train \
    --output_dir=./resources/detections/MOT16_train


I used the following code: python tools/generate_detections.py --model=resources/networks/mars-small128.pb --mot_dir=./MOT16/train --output_dir=./resources/detections/MOT16_train



##4 Converting the numpy_array(.npy) file as per Multiple Object Tracking Evaluation Metrics standards, and saving it in the results folder as txt files.

python evaluate_motchallenge.py --mot_dir=C:\Users\arnab\Documents\extd_deepsort\deep_sort\MOT16\train --detection_dir=C:\Users\arnab\Documents\extd_deepsort\deep_sort\resources\detections\MOT16_POI_train

##5 Evaluating the metrics using py-mot metrics (https://github.com/cheind/py-motmetrics)

Git Clone this repository https://github.com/cheind/py-motmetrics and inside the directory make 2 folders 1) GT for Groundtruth files and 2) TEST for testing files

For GT copy everything inside MOT16/train directory
For TEST copy all the Txt files from deep sort/results folder and run this command (replace <GT directory> and <TEST directory> with the actual path)

python -m motmetrics.apps.eval_motchallenge <GT directory> <TEST directory>

This is how I used:
python -m motmetrics.apps.eval_motchallenge C:\Users\arnab\Documents\deep_sort\deep_sort-master\py-motmetrics-develop\GT C:\Users\arnab\Documents\deep_sort\deep_sort-master\results

This will show the results.

