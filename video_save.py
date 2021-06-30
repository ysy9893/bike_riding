######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from nms import NMS
from motpy import Detection, MultiObjectTracker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import pandas as pd 

#Create a new figure
fig=plt.figure()
ax2=fig.add_subplot(1,1,1)







# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--save',default='./output.mp4')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
save=args.save

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv2.VideoWriter(save, fourcc, fps, (int(imW), int(imH)))

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()#Return clock cycle per second

#Initialize Tracker 
tracker=MultiObjectTracker(dt=0.1)

#Center of bbox
dict_idx={}
compare_idx={}


while(video.isOpened()):
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    
    ax2.set_xlim(0, imW)
    ax2.set_ylim(0, imH*2)
    


    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    # Non Maximum Suppression 
    boxes,scores,classes=NMS(boxes,classes,scores,0.5,imH,imW)
    
    ## Configuratino for Tracking 
    boxes=np.array(boxes)#current col order is [ymin,xmin,ymax,xmax]
    # Change the order of cols to [xmin,ymin,xmax,ymax] which is suitable feature of feed for tracker
    xmin=boxes[:,1]*imW
    xmin[xmin<1]=1
    xmin=xmin.reshape((-1,1))
    ymin=boxes[:,0]*imH
    ymin[ymin<1]=1
    ymin=ymin.reshape((-1,1))
    xmax=boxes[:,3]*imW
    xmax[xmax>imW]=imW
    xmax=xmax.reshape((-1,1))
    ymax=boxes[:,2]*imH
    ymax[ymax>imH]=imH
    ymax=ymax.reshape((-1,1))
    
    boxes=np.concatenate((xmin,ymin,xmax,ymax),axis=1)
    boxes=[i for idx,i in enumerate(boxes) if scores[idx]>min_conf_threshold and scores[idx]<=1.0 and classes[idx]==2]
    classes_temp=classes

    classes=[i for idx,i in enumerate(classes) if scores[idx]>min_conf_threshold and scores[idx]<=1.0 and i==2]
    scores=[i for idx,i in enumerate(scores) if i >min_conf_threshold and i<=1.0 and classes_temp[idx]==2]
    
    ##Tracking 
    detections=[Detection(box=bbox,score=sc,cl=cl) for bbox, sc, cl in zip(boxes,scores,classes)]
    tracker.step(detections)
    tracks=tracker.active_tracks()
    
    ##Dictionary containig previous frame's objects
    compare_idx=dict_idx
    dict_idx={}
    ##color maps for each tracked object
    color={0:(0,0,0),1:(255,255,0),2:(0,0,255),3:(255,192,203),4:(0,255,255),5:(255,0,0),6:(0,255,0)}
    c=0

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for track in tracks:
        
        
        if True:

            ###Get bounding box coordinates and draw box
            
            xmin = int(track.box[0])
            ymin = int(track.box[1])
            xmax = int(track.box[2])
            ymax = int(track.box[3])
            w=xmax-xmin
            h=ymax-ymin
            
            bx_id=track.id
            x_center=(xmin+xmax)/2
            y_center=(ymin+ymax)/2
            area=w*h
            
            ##Matching bbox btw 2 diff frames
            #dict.format=>[xcenter,ycenter,colormap,bbox_area]
            
            if len(compare_idx)==0:
                dict_idx[bx_id]=[x_center,y_center,c,area]
                
            else:
                
                if bx_id in compare_idx.keys():
                    #area gets bigger 
                   dict_idx[bx_id]=[x_center,y_center,compare_idx[bx_id][2],area]                        
                        
                        
                #newly detected bbox         
                else:
                    dict_idx[bx_id]=[x_center,y_center,c,area]

            
            if c==6:
                c=0
            c+=1
        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color[dict_idx[bx_id][2]], 2)

            # Draw label
            object_name = labels[int(track.cl)]
            label = '%s: %s%%' % (object_name, str(int(dict_idx[bx_id][0]))) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    
    colormap={0:'k',1:'aquamarine',2:'r',3:'m',4:'y',5:'b',6:'g'}
    if len(compare_idx) !=0:
        #Get x,y,color,area values
        
        temp_dict=list(dict_idx.values())
        temp_dict =np.float32(temp_dict)
        x=temp_dict[:,0]
        y=temp_dict[:,1]*2

        cmap=temp_dict[:,2]
        cmap=[colormap[i] for i in cmap]
        size=temp_dict[:,3]/(imW*imH)*100000


        ax2.scatter(x,y,size,cmap,marker='o')

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    out.write(frame)

   
    #Press 'ESC' to quit 
    if cv2.waitKey(30) &0xff==27:
        break

    plt.show(block=False)
    plt.pause(0.01)
    plt.cla()

# Clean up
video.realse()
out.realse()
cv2.destroyAllWindows()



