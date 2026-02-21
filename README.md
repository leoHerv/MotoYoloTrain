# MotoYoloTrain

 Project to train yolo models for motocross plate detection.

## Goal 

1. Make a pipeline to train yolo11 obb models (for plate detection and number detection).
2. Make a UI for the training

### TODOS

- Fix out of image labels
- Optimize performance

## Annotation software 

For the annotation, the project [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) is used.


## Install 

Run the command `pip install -r requirements.txt` to install all the packages needed for this project.

Also, **CUDA 12.4** and **cuDNN 9.3.0** is needed to make Pytorch work with an nvidia GPU.

### Link :  
CUDA 12.4 : https://developer.nvidia.com/cuda-12-4-0-download-archive
cuDNN last version (9.3.0) : https://developer.nvidia.com/cudnn-downloads

The table to see which version of CUDA you need to make Pytorch work :  
https://pytorch.org/get-started/locally/

## Run

Create the `.env` file and specify the paths needed for the program.  
`.envExample` is example of what you need to specify in the `.env`.  

### Specs for training

`Laptop RTX 4070` and `Ryzen 7 7745HX`

## Infos 

This project is mainly vibe coded with `Gemini code Assist`.
For this project, high performance is not needed.