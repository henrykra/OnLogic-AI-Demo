# AI Demo Documentation

Author: Henry Kraessig, AI Intern

## Intro

This document should describe the complete pipeline that I used to create the demo for the 9/14/2024 airfield show. The goal behind this demo was to make it expandable, to hopefully be able to reuse the demo for future trade shows or to be able to expand it to make it do more complex tasks. At its core, the demo is meant to be an example of how to use AI/Computer Vision in a manufacturing setting to improve quality and decrease the occurrence of defects. We decided to have this demo discern between 4 types of lego figures using an object detection model, but this can be expanded to do more detailed defect detection or more complex computer vision implementations. The challenge with expanding this demo in the future is that I’m the only one who worked on the coding, and I’m just an intern, which means, by the time the demo is updated or used again, I might be long gone (hopefully not, I’d love to be hired back). This document should describe how the demo was created, which should make it easy to change in the future with the minimal overhaul. Or, if my process could be improved, overhaul as you wish. 


## Model Creation/Training

The object detection model that runs the demo was created with the Ultralytics Python library, and transfer learning from the pre-trained YOLOv8s (small) model. The model was trained on 320 pictures of the lego figures for the demo. I used the helper script take_pictures.py to take these pictures and save them to a folder with a common name scheme to prepare for labeling. Once I had enough pictures, I imported them into label-studio and labeled them all, very tediously, by hand. 

The ultralytics model needs images and labels in the yolo .txt format. Label-studio supports this export format, which is helpful for creating a final data folder which is fed to the model for training. However, I believe there is an exporting limit on the non-enterprise version of label-studio, so I wasn’t able to export all 320 of my training images in yolo format. Instead, I exported in csv format and used the csv_to_yolo.py helper script to create yolo .txt label files for each image. 

If you use the csv_to_yolo.py script, you will make a folder titled “labels” containing a .txt for every image mentioned in the label-studio csv output. Label files will be named the same as their corresponding image, but with a .txt extension. From here, you need to create the folder structure required for yolo model training. Info about this file format is here. In general, what you need to do is create a dataset folder, named whatever you like, which contains an images and labels folder. The images folder should be the raw images (output from take_pictures.py), and the labels folder should be the folder created from csv_to_yolo.py.

```
├── data_folder (to be named by you)
│   ├── Images
│   │   ├── img0.png
│   │   ├── img1.png
│   │   ├── ...
│   ├── Labels
│   │   ├── img0.txt
│   │   ├── img0.txt
│   │   ├── ...
```

After this folder structure is created, create a file called classes.txt within the data folder that contains a list of the class names. It should just contain the names of the classes, in the same order that they should appear in future label maps, separated by newlines characters. 

Then, run the train_test_split.py file to randomly create training and validation partitions in the data, as well as create a data.yaml file for model training. In train_test_split.py, you can update the path variables to direct to the data folder, and change the training set size. Once this file is run, the labels and images folders should have train and val subfolders containing subsets of the data. Any images that don’t have a corresponding label will remain in the images folder, and can be removed if you like. The data.yaml file will be created, and contain paths to the training and validation data, as well as a path to itself, which is set up to work when imported into google drive (for training on colab). You will need to edit this path to match how you plan to store this folder in drive. 

To train the model, I used google colab, and imported the entire data folder into my drive. To train and export the model, only a couple lines of code are needed. (Make sure the colab is using a gpu runtime, or training will be very slow).

```python
%pip install ultralytics
from ultralytics import YOLO
model = YOLO('path_to_yolov8.pt')
model.train(data='path_to_data.yaml', epochs=15, imgsz=640, val=True)
model.export(format='openvino', dynamic=True, half=True)
```

Download the best.pt and best_openvino_model files from the runs/train/detect/weights folder that was created by the training call. I collect the pytorch and openvino model formats because the open_vino format optimizes inference for intel hardware, and has support for intel iGPUs. In the future, one could ignore the .pt file and only use the openvino format for the entire detection process (preprocessing, inference, post-processing), but I thought that could be too time-consuming for the demo deadline, so I used the ultralytics pre and post processing, which uses the CPU. 



## Web UI

The demo interface runs off of a Flask app, which functions as a live feed for the camera, and a way to interact with the air compressor so that the model can react to the detection results in the real world. By trade, I am not a web designer, so I think this is the part of the demo that should probably be optimized. 

The app is run on an HX401 in an anaconda environment, using Python 3.10. The environment requires a couple packages, which include but are not limited to:

* Flask
* PyTorch
* Ultralytics
* Openvino
* OpenCV (cv2)

There are only a couple of important files for the web app. Interface.py contains all of the backend logic that collects user input from a form, and decides what to do with object detection model output. The app has a main loop which executes every tenth of a second (10fps). Simply, the loop:
1. Collects the current image from the camera
2. Feeds it into an imported openvino detection model
3. Removes detections that occur in a predefined “deadzone”
4. This deadzone attempts to avoid detections being picked up from the slide where the lego figures go after they’re removed from the turntable
5. Parses the detection results to determine if there are any figures from the users selections past a certain “detection threshold”
6. Adds times to operate the air compressor to a queue
7. Checks the queue to see if it's time to operate the compressor
8. Plots the detection boxes on the image from the camera
9. Encodes the plot to a html byte stream response and displays it on the interface

	
The step of the loop that took me the most time was the way to get the backend to “understand” the detection output. The way it is setup as of now, the app ignores any detection boxes where the bottom y coordinate is above 375 pixels (nearly the whole way down the 480px feed). This happens because at this point the detections are much more reliable due to the autofocus of the camera, and once the figures get this close to the camera, the bottom y-value of the detection box moves down the screen with approximately linear speed. This is the assumption that drives the solution I have to deciding when to fire the air compressor. 

Because at this distance from the camera, the figure is moving nearly linearly with constant speed towards the air compressor, and the detection is moving linearly across the screen, there is a linear function that approximates the time it will take for the figure to reach the line of fire based on the pixel coordinates of the detection box. I’ve approximated this function using some hand-timing and guess and checking, so it can be improved in the future. 

The function mentioned above is crucial for the model to be able to identify each figure as it arrives on the turntable as individuals. The model doesn’t have any “object permanence”, so as each figure arrives close to the camera, it can’t discern frame-to-frame which object is the same as from the last frame. It needs to be able to identify and track figures frame-to-frame to avoid executing multiple air-blasts for the same figure, but be able to perform two blasts when there are two figures close together. 

The way it does this is, for each frame of model output, the model checks to see if there are any figures that must be removed within the detection threshold. If there are, it finds the y-coordinate of the detection box and calculates the amount of time it will take for the figure to reach the air nozzle. It adds that calculated time to a queue. At each frame, the model will check the first element of the queue (the earliest) and see if the current device time is past the time value entered. If it is, then the air will fire, and the figure will (hopefully) be removed. To avoid adding multiple air blasts to the queue for a single figure, the logic will check to see if the calculated time to operate the air (based on the y-coordinate of the detection box) is within .1 seconds of any of the times within the queue. If it isn’t, the time is added to the queue, which means this is the first time this figure has been detected. If there is a time in the queue within this error time, there will be no air blasts added to the queue, because the model knows that either this detection is the same as a previous detection, or there is another figure close enough to this figure that two air blasts aren’t needed. 

This may not be the most ideal/industry standard solution to box tracking, but it has been working well, avoiding double blasts while still being able to distinguish between two close-together figures. In the future, this logic can be updated to have an expanded detection threshold zone, which should allow for more frames to be used to determine what to do with any given detection, and reduce potential error. The function will lose its linearity, but because of the predictable nature of the constant-speed turntable, a new function can be created that would work just as well. 

The other file of note is template.html which contains the live detection feed and form for user input, and is a very rudimentary html file. There are some css rules to hopefully organize it intuitively for the end user. The app hosting should be changed in the future, but I was not able to change it within the deadline of the airfield show. 

The web UI is shown on a touch panel, which opens the app in chrome kiosk mode, which stops the user from being able to exit out of the app without a keyboard. Operation of the demo app and potential failure points are described in the Demo Instructions. 

	
