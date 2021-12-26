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
-- random_horizontal_flip
-- random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }

python edit_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt

2. <enhance_config.py> is used to create enhanced configuration file for Tensorflow Object Detection API (pipeline_new.config) under ./training/enhance/
-- random_horizontal_flip
-- random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
-- random_adjust_brightness {
      max_delta: 0.2
    }
-- random_rgb_to_gray {
      probability: 0.2
    }
-- random_adjust_contrast {
      min_delta: 0.6
      max_delta: 0.9
    }

python enhance_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt --brightness 0.5

3. <change_config.py> is used to create another enhanced configuration file with different optimizer configuration for Tensorflow Object Detection API (pipeline_new.config) under ./training/change/
-- random_horizontal_flip
-- random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
-- random_adjust_brightness {
      max_delta: 0.2
    }
-- random_rgb_to_gray {
      probability: 0.2
    }
-- random_adjust_contrast {
      min_delta: 0.6
      max_delta: 0.9
    }
--optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.015
          total_steps: 25000
          warmup_learning_rate: 0.013333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }

python change_config.py --train_dir ./training/ --eval_dir ./validation/ --batch_size 4 --checkpoint ./pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./label_map.pbtxt --brightness 0.2 --learning_rate 0.015

### Training

model_main_tf2.py is used to train models

python ./experiments/model_main_tf2.py --model_dir=./training/reference/ --pipeline_config_path=./training/reference/pipeline_new.config

python ./experiments/model_main_tf2.py --model_dir=./training/enhance/ --pipeline_config_path=./training/enhance/pipeline_new.config

python ./experiments/model_main_tf2.py --model_dir=./training/change/ --pipeline_config_path=./training/change/pipeline_new.config


### Enhancement

Explore augmentations.ipynb is used to explore potentials to improve models

jupyter notebook

1. An Example of <edit_config.py>

![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/referenceexample.png?raw=true)

2. An Example of <enhance_config.py>

![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/enhanceexample.png?raw=true)

3. An Example of <change_config.py>

![alt text](https://github.com/vickyting0910/objectdetection/blob/main/images/changeexample.png?raw=true)

### Evaluation

model_main_tf2.py is also used to validate models. 

python ./experiments/model_main_tf2.py --model_dir ./training/reference/ --pipeline_config_path ./training/reference/pipeline_new.config --checkpoint_dir ./training/reference/

python ./experiments/model_main_tf2.py --model_dir=./training/enhance/ --pipeline_config_path=./training/enhance/pipeline_new.config --checkpoint_dir=./training/enhance/

python ./experiments/model_main_tf2.py --model_dir=./training/change/ --pipeline_config_path=./training/change/pipeline_new.config --checkpoint_dir=./training/change/

### Export Models 

exporter_main_v2.py is used to save models for future uses. 

python ./experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/reference/pipeline_new.config --trained_checkpoint_dir training/reference --output_directory training/reference/exported_model/

python ./experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/enhance/pipeline_new.config --trained_checkpoint_dir training/enhance --output_directory training/enhance/exported_model/

python ./experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/change/pipeline_new.config --trained_checkpoint_dir training/change --output_directory training/change/exported_model/

### Inference

inference_video.py is used to inference the new image. In this case, segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord is used. 

python inference_video.py --labelmap_path label_map.pbtxt --model_path training/reference/exported_model/saved_model --tf_record_path testing/segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord --config_path training/reference/pipeline_new.config --output_path training/reference/animation.mp4

<video src='https://github.com/vickyting0910/objectdetection/blob/main/images/animation_reference.mp4' width=180/>

python inference_video.py --labelmap_path label_map.pbtxt --model_path training/enhance/exported_model/saved_model --tf_record_path testing/segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord --config_path training/enhance/pipeline_new.config --output_path training/enhance/animation.mp4

<video src='https://github.com/vickyting0910/objectdetection/blob/main/images/animation_enhance.mp4' width=180/>

python inference_video.py --labelmap_path label_map.pbtxt --model_path training/change/exported_model/saved_model --tf_record_path testing/segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord --config_path training/change/pipeline_new.config --output_path training/change/animation.mp4

<video src='https://github.com/vickyting0910/objectdetection/blob/main/images/animation_change.mp4' width=180/>

