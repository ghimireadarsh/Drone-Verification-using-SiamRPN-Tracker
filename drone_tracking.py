# This code runs the siamRPN for drone tracking and verification

import os
import argparse
import math
import numpy as np
import cv2
import torch
import imutils
import time
from imutils.video import VideoStream
from imutils.video import FPS
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

#os.environ['KMP_DUPLICATE_LIB_OK']='True' # This is required for macos, otherwise without it there will be hardware issue

torch.set_num_threads(1)
torch.cuda.empty_cache() # clearing the gpu cache
parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('-c', '--config', type=str, help='config file')
parser.add_argument('-s', '--snapshot', type=str, help='model name')
parser.add_argument('-v', '--video', default='', type=str, help='videos or image files')
args = vars(parser.parse_args())

def main():
    # load config
    cfg.merge_from_file(args["config"])
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args["snapshot"],
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)
    # print(model)
    # build tracker
    tracker = build_tracker(model)
    #print(tracker)

    # initialize the bounding box coordinates of the object we are going to track
    gt_bbox = None
    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])

    success = False
    drone_x = [] # drone movement in x before control signal is sent
    drone_y = [] # drone movement in y before control signal is sent
    after_command_x = [] # drone movement in x after control signal is sent
    after_command_y = [] # drone movement in y after control signal is sent
    limit_datapoint = 50 # count of number of datapoints for best fit generation
    control_data = None # control signal developed based on the best fit generated
    verification_threshold = 20 # in degrees for looking out how much angle does the drone changes
    drone_success = [] # to store count of drone successful behavior
    command_exec_time = True # to track drone within certain time after sending control signal
    execution_deadline = 100 # count of number of drone successful behavior
    drone_status = "UNKNOWN"
    color = (255, 0, 0)
    control_signal_status = "Generating" # by default
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        # resize the frame (so we can process it faster) and grab the
        # frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if gt_bbox is not None:
            outputs = tracker.track(frame)
            # print(outputs)
            box = outputs["bbox"]
            if outputs['best_score'] > 0.8:
                success = True
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            else:
                success = False
            # update the FPS counter
            fps.update()
            fps.stop()
            info = [
                ("Tracker", "Siamese RPN"),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
                ("Drone Status", "{}".format(drone_status)),
                ("Control Signal", "{}".format(control_signal_status)),
            ]
            # loop over the info tuples and draw them on frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if success and control_data is None:
            if len(drone_x) < limit_datapoint:
                    drone_x.append(box[0]+box[2]/2)
                    drone_y.append(box[1]+box[3]/2)
            else:
                print("Total points : {}".format(len(drone_x)))
                # control signal function
                best_fit = np.polyfit(drone_x, drone_y, deg=1)  # returns (m, b)
                (control_data, control_data_1) = control_signal(best_fit)
                print("Best fit : {}".format(best_fit))
                print("Theta : {}".format(math.degrees(math.atan(best_fit[0]))))
                print("Control signal data received : {}".format(control_data))
                control_signal_status = "Complete"

        if success and control_data is not None and command_exec_time:
            drone_status = "VERIFYING"
            if len(after_command_x) < limit_datapoint:
                after_command_x.append(box[0] + box[2] / 2)
                after_command_y.append(box[1] + box[3] / 2)
            else:
                after_command_best_fit = np.polyfit(after_command_x, after_command_y, deg=1)  # returns (m, b)
                after_command_theta = math.degrees(math.atan(after_command_best_fit[0]))
                # print(after_command_theta)
                diff = abs(abs(after_command_theta) - abs(control_data))
                diff_1 = abs(abs(after_command_theta) - abs(control_data_1))
                # here time limit to be added later
                if diff < verification_threshold or diff_1 < verification_threshold:
                    # print("*************Good Drone*************")
                    # print("Difference between the required angle is {}".format(diff))
                    drone_success.append(1)
                else:
                    # print("**************Bad Drone*************")
                    # print("Not with in the threshold {}".format(diff))
                    drone_success.append(0)
                    after_command_x.pop(0)
                    after_command_y.pop(0)
                    after_command_x.append(box[0]+box[2]/2)
                    after_command_y.append(box[1]+box[3]/2)

        if len(drone_success) > execution_deadline:
            command_exec_time = False
            if drone_success.count(1)/len(drone_success) >= 0.2:
                print("***************Good drone****************")
                drone_status = "GOOD"
                color = (0, 255, 0)
            else:
                print("***************Bad drone*****************")
                drone_status = "BAD"
                color = (0, 0, 255)

        if key == ord("s"):
            # select the bounding box of the object we want to track
            # (make sure you press ENTER or SPACE after selecting the ROI)
            gt_bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # print(gt_bbox)
            tracker.init(frame, gt_bbox)
            print("Tracker Initialization success")
            fps = FPS().start()
            # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        vs.stop()
    # otherwise, release the file pointer
    else:
        vs.release()
    # close all windows
    cv2.destroyAllWindows()

def control_signal(best_fit):
    theta = math.degrees(math.atan(best_fit[0]))
    control_data = theta + 90
    control_data_1 = theta - 90
    start = time.time()
    while(time.time()-start < 0.3):
        print("Sending control signal, theta {}".format(control_data))
        time.sleep(0.1)
    return (control_data, control_data_1) # for positive and negative angle
if __name__ == '__main__':
    main()