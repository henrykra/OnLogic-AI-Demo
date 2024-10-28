from flask import Flask, render_template, Response, request
import cv2
import time
from ultralytics import YOLO
import openvino as ov
import torch
import serial
from serial.tools import list_ports as system_ports
#TODO: import from livedetection.py instead of copying fucntions

app = Flask(__name__)

# This variable stores the selections for each of the four lego classes. 
# If the user selects one of the checkboxes on the interface, the corresponding index will turn to true. 
# The indicies if this list correspond to the entries in the label map aswell as yolo output
class_inputs = [False, False, False, False]

label_map = {
    0: "Captain Antilles",
    1: "Imperial Stormtrooper",
    2: "Rebel Trooper",
    3: "Clone Trooper"
}

CONFIDENCE_LEVEL = .8
WAIT_TIME_CONSTANT = .25
IOU_LEVEL = .7
FPS_LIMIT = 10
MCU_VID_PID = '353F:A104'
CMD_OFF = 'dio set DIO0 0 false'
CMD_ON = 'dio set DIO0 0 true'

"""
Setup for DIO
"""
def get_device_port(dev_id, location=None):
    """Scan and return the port of the target device."""
    all_ports = system_ports.comports()
    for port in sorted(all_ports):
        print(port.hwid)
        if dev_id in port.hwid:
            if location and location in port.location:
                print(f'Port: {port}\nPort Location: {port.location}\nHardware ID: {port.hwid}\nDevice: {port.device}')
                print('*'*15)
                return port.device
    return None

mgmt_port = get_device_port(MCU_VID_PID, "3-8:1.0")
ser = serial.Serial(mgmt_port, 9600)



# This is route for the interface page. It accepts get and post requests, and behaves differently for each. 
@app.route("/", methods=['POST', 'GET'])
def interface_func():
    # By default, the interface does not have detection on, so detect is set to false
    detect = False
    # This route needs to change the class_inputs list to reflect the user selection, so it accesses the global variable. 
    global class_inputs

    # The request method is post whenever the "detect" button is pressed. The next lines allow the backend to access the user selections. 
    if request.method == 'POST':
        # these are the ids of the checkboxes
        class_input_ids = ['input1', 'input2', 'input3', 'input4']
        # change the class_inputs global variable to reflect checkbox states
        class_inputs = [id in request.form.keys() for id in class_input_ids]
        
        # when the detect button is pressed, detection is true
        detect = True
    else: 
        # if the page recieves a get request, detection stays false and the class_inputs list can return to its default state
        class_inputs = [False, False, False, False]

    # renders the interface.html template and passes the user selections(to change the border of the label elements)
    # and the detect variable, which determines which feed to access (detection or raw camera).
    return render_template('interface.html', detect=detect, classes=class_inputs)

# global function that initializes pretrained yolo model from openvino and .pt format
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
    
    res = model(input, iou=IOU_LEVEL, conf=CONFIDENCE_LEVEL, verbose=False)
    return res[0]

# funtion to draw bounding boxes over one frame
def draw_boxes(result, class_inputs):
    """
    Function that draws a single frame of bounding boxes given a result
    
    Args:
        result: single ultralytics result object
        class_inputs: list of user selections
        
    Returns: 
        Image array with drawn on bounding boxes (BGR)"""
    
    # tensor in form ([x1, y1, x2, y2, conf, class], []...)
    data = result.boxes.data

    img = result.orig_img
    line_thickness = 2

    for box in data:
        pred_class = int(box[-1])
        # check if the predicted class has a value of true in class inputs
        # meaning that it should be marked red for removal
        if class_inputs[pred_class]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3])) # coordinates
        # draw rectangle
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return img


def activate_air(ser):
    ser.write(b'\r\n')

    ser.write(CMD_ON.encode())
    ser.write(b'\r\n')
    print(ser.read(ser.inWaiting()))


    time.sleep(0.03)
    ser.write(CMD_OFF.encode())
    ser.write(b'\r\n')


model = initialize_model()
cam = cv2.VideoCapture(0) 

# Generator function which provides a byte stream of the camera output
def gather_feed():
    t = time.time()
    while True:
        _, img = cam.read()

        if FPS_LIMIT:
            time.sleep(max(0, 1 / FPS_LIMIT - (time.time() - t)))
        t = time.time()

        _, frame = cv2.imencode('.jpg', img)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

# function which decides what to do with the yolo output, and operates the air compressor
def parse_result(result, class_inputs, queue):
    """Function for determining whether or not to activate air given a result
    
    Args:
        result: single ultralytics result object
        class_inputs: list of user selections
        queue: the list of requests for air, which is a queue of future timestamps, which is constantly checked
               to see if the first element(earliest) contains a time later than the current device time (time to fire the air)
               
    Returns:
        nothing, just makes http requests based on output"""
    
    THRESHOLD = 375 # pixel value of detection threshold
    # Once the bottom of the detection box reaches below the threshold pixel, air is queued (if the class of the detection
    # is one to be removed)

    removal_boxes = [] # list of bad classes past the "Threshold"

    # tensor of [[x1, y1, x2, y2, conf, class], [], ...]
    boxes = result.boxes.data

    for box in boxes:
        if float(box[3]) >= THRESHOLD and float(box[3]) <= THRESHOLD + 100 and class_inputs[int(box[-1])]:
            # if there is a box below the threshold but above the very bottom of the viewport, and if its one of the
            # selected classes in `class_inputs`, add the box to the removal boxes list.
            removal_boxes.append(box)

    # for each box that needs to be removed
    for box in removal_boxes:
        # calculates the time at which the air needs to fire. The time is based on using rough estimates of scalar values
        # which predict (linearly) the time the figure will take to be in the line of fire based on its pixel locations
        queue_time = time.time() + WAIT_TIME_CONSTANT - ((float(box[3]) - 375) / 50) * .17
        
        # Check the queue if there is already an air blast scheduled for a time close to the time calculated above
        # if there is, don't add a second blast to the queue
        add = True
        for t in queue:
            if queue_time - t <= .1: # .1 seconds is the "error threshold", the time two blasts should be apart to be added individually
                add = False

        if add:
            queue.append(queue_time)

    # if there are any times in the queue, check if the current time is after the first time in the list
    if queue:
        t = queue[0]
        if time.time() >= t:
            # if its time to activate, make a call to the controller
            print('Air Activated')
            activate_air(ser)
            queue.pop(0)
            # remove the time once the call is made
        
    # return the queue for subsequent calls
    return queue


# We don't want the model to be distracted by detections that arent on the track
def result_deadzone(result):
    data = result.boxes.data

    removal_indicies = []

    def pt_delete(tensor, indicies):
        mask = torch.ones(tensor.size()[0], dtype=torch.bool)
        mask[indicies] = False
        return tensor[mask]
    
    
    for i, box in enumerate(data):
        y = float(box[1])
        x = float(box[0])
        


        if y <= 200 - x * .6: # rough line segment that represents out of track area
            removal_indicies.append(i)

    if removal_indicies:
        result.boxes.data = pt_delete(data, removal_indicies)
    
    return result
            

# Generator which returns the byte stream of object detection results from the camera
# calls the detect, draw_boxes, and parse_result functions
def gather_img(class_inputs, blast_air):
    # buffer = []
    t = time.time()
    air_queue = []

    while True:
        # get an image from the camera
        _, img = cam.read()
        # run custom object detection
        result = detect(model, img)

        result = result_deadzone(result)

        # limit the detection to 10 fps, to avoid 100% gpu utilization
        if FPS_LIMIT:
            time.sleep(max(0, 1 / FPS_LIMIT - (time.time() - t)))

        t = time.time()

        # buffer.append(result)
        # if len(buffer) > FRAME_BUFFER:
        #     buffer.pop(0)

        # get air queue from the parse result function

        if blast_air:        
            air_queue = parse_result(result, class_inputs, air_queue)
            

        # get the image with boxes from the draw_boxes function
        img = draw_boxes(result, class_inputs)
        # img = result.plot()

        # encode image and yield byte stream
        _, frame = cv2.imencode('.jpg', img)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


# route for the live feed, is embedded in the interface when it recieves a get request (no detection)
@app.route("/feed")
def feed():
    return Response(gather_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# route for live detection feed, embedded in the interface when it recieves a post request (detection)
@app.route('/feed/detect')
def feed_detect():
    return Response(gather_img(class_inputs, True), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(threaded=True)
