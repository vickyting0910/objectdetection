# Object Detection in an Urban Environment

## Objectives

This project aims for identifying objects using TensorFlow object detection API, especially on cars,pedestrians, cyclists. The idea is fit rectangular bounding boxes on the images with objects in different conditions. Here are 10 sample images.  

![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/samples.png?raw=true)

## Data

This project uses the data from the [Waymo Open dataset](https://waymo.com/open/). 

## Pipeline

### Setup Google Cloud Credentials

1. Download the file and extract from https://cloud.google.com/sdk/docs/install

2. Run ./google-cloud-sdk/bin/gcloud init

3. Log in by gcloud auth login

4. Check in the bucket: https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/


### Download and Pre-Processing Data

download_process.py is used to download files based on filenames.txt (100 files) and convert from Waymo tf record into a tf api tf record and example.

python download_process.py --data_dir ./data

### Exploratory Data Analysis

<Exploratory Data Analysis.ipynb> shows the visualization results of data. 

jupyter notebook

1. Distribution of Classes for 500 Random Images
![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/distribution.png?raw=true)

2. Percentage of Car Class over All Class
![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/car.png?raw=true)

3. Percentage of Pedestrian Class over All Class
![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/ped.png?raw=true)

4. Percentage of Cyclist Class over All Class
![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/cyc.png?raw=true)

### Split Data

<create_splits.py> is used to split 100 files into training, testing and validation datasets, placed in 3 different folders under ./training/ ./testing/ ./validation/.  

python create_splits.py --source ./data/processed --train_prop 0.75 --test_prop 0.1 --valid_prop 0.15

### Download the Pretrained Model

<download_pretrain.py> is used to download and extra the pretrained model

python download_pretrain.py

### Set up config file

1. <edit_config.py> is used to create configuration file for Tensorflow Object Detection API (pipeline_new.config) under ./training/reference/

python edit_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt

2. <enhance_config.py> is used to create enhanced configuration file for Tensorflow Object Detection API (pipeline_new.config) under ./training/enhance/

python enhance_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt --brightness 0.5

python change_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt --brightness 0.2 --learning_rate 0.015

### Training

model_main_tf2.py is used to train models

python ./experiments/model_main_tf2.py --model_dir=./training/reference/ --pipeline_config_path=./training/reference/pipeline_new.config

python ./experiments/model_main_tf2.py --model_dir=./training/enhance/ --pipeline_config_path=./training/enhance/pipeline_new.config

python ./experiments/model_main_tf2.py --model_dir=./training/change/ --pipeline_config_path=./training/change/pipeline_new.config


### Enhancement

Explore augmentations.ipynb is used to explore potentials to improve models

jupyter notebook

### Evaluation

model_main_tf2.py is also used to validate models. 

python ./experiments/model_main_tf2.py --model_dir=./training/reference/ --pipeline_config_path=./training/reference/pipeline_new.config --checkpoint_dir=./training/reference/

python ./experiments/model_main_tf2.py --model_dir=./training/enhance/ --pipeline_config_path=./training/enhance/pipeline_new.config --checkpoint_dir=./training/enhance/

python ./experiments/model_main_tf2.py --model_dir=./training/change/ --pipeline_config_path=./training/change/pipeline_new.config --checkpoint_dir=./training/change/

### Animation

exporter_main_v2.py is used to 

python ./experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/experiment0/pipeline.config --trained_checkpoint_dir training/experiment0 --output_directory training/experiment0/exported_model/

### Inference

inference_video.py is used to 

python inference_video.py -labelmap_path label_map.pbtxt --model_path training/experiment0/exported_model/saved_model --tf_record_path /home/workspace/data/test/tf.record --config_path training/experiment0/pipeline_new.config --output_path animation.mp4

