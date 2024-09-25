from ultralytics import YOLO
import cv2 as cv
import numpy as np
import openvino as ov
import torch
import time

# TODO: allow for override args (conf) and look into stream=True predicting
# TODO: include frame rate data
# TODO: include prediction data for real-world interaction
def initialize_model():
    # initialize trained ultralytics yolo model from .pt file
    model = YOLO('models/best.pt')
    model.predictor = model._smart_load('predictor')()

    # import openvino model for intel optimized inference
    # ultralytics model is still used for pre and post-processing at the moment, which is why both are needed
    core = ov.Core()
    ov_model = core.read_model('models/best_openvino_model/best.xml')
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    ov_compiled_model = core.compile_model(ov_model, 'GPU', ov_config)


    def infer(*args):
        result = ov_compiled_model(args)
        return torch.from_numpy(result[0])
    

    # use ov inference
    model.predictor.inference = infer
    # model.predictor.model.pt = False

    return model


# detects from a single frame
def detect(model, input): # see top todo
    
    res = model(input)
    return res[0].plot() # returns BGR array with predicted bounding boxes


def run_live_detection():
    model = initialize_model()
    
    cap = cv.VideoCapture(0)

    # only for a couple frames
    for _ in range(200):
        ret, frame = cap.read()

        if not ret:
            print('Frame not found. Exiting.')
            break


        pred = detect(model, frame)
        
        cv.imshow('frame', pred)
        # limit frame rate
        time.sleep(.1)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows


if __name__ == '__main__': 
    run_live_detection()