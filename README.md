# DeepDarts

Code for the CVSports 2021 paper: [DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera](https://arxiv.org/abs/2105.09880)

## Prerequisites
Python 3.5-3.8, CUDA >= 10.1, cuDNN >= 7.6

## Setup
1. [Install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a new conda environment with Python 3.7: ```$ conda create -n deep-darts python==3.7```. Activate the environment: ```$ conda activate deep-darts```
4. Clone this repo: ```$ git clone https://github.com/wmcnally/deep-darts.git```
5. Go into the directory and install the dependencies: ```$ cd deep-darts && pip install -r requirements.txt```
6. Download ```images.zip``` from [IEEE Dataport](https://ieee-dataport.org/open-access/deepdarts-dataset) 
   and extract in the ```dataset``` directory. Crop the images: ```$ python crop_images.py --size 800```. This step could
   take a while. Alternatively, you can download the 800x800 cropped images directly from IEEE Dataport. 
   If you choose this option, extract ```cropped_images.zip``` in the ```dataset``` directory.
8. Download ```models.zip``` from IEEE Dataport and extract in the main directory.

## Validation / Testing
To test the Dataset 1 model:\
```$ python predict.py --cfg deepdarts_d1 --split test```


To test the Dataset 2 model and write the prediction images: \
```$ python predict.py --cfg deepdarts_d2 --split test --write```


## Training
To train the Dataset 1 model:\
```$ python train.py --cfg deepdarts_d1```

To train the Dataset 2 model:\
```$ python train.py --cfg deepdarts_d2```

You may need to adjust the batch sizes to fit your total GPU memory. The default batch sizes are for 24 GB total GPU memory.

## Sample Test Predictions

Dataset 1:\
![alt text](./d1_pred.JPG)

Dataset 2:\
![alt text](./d2_pred.JPG)



