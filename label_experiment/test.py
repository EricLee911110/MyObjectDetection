from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import os
import cv2
import subprocess

INPUT_PATH = 'input_images/'
OUTPUT_PATH = 'output_images/'
MIN_CONFIDENCE = 0.7
MODEL_TYPE = "frcnn-mobilenet"
object_count = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = pickle.loads(open('../coco_classes.pickle', "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

model = MODELS[MODEL_TYPE](pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()


for x in os.listdir(INPUT_PATH):
    path_input_image = INPUT_PATH + x
    output_image_name = x.split('.')[0] + "_fin." + x.split('.')[1]
    path_output_image = OUTPUT_PATH + output_image_name
    
    image = cv2.imread(path_input_image)
    orig = image.copy()

    #image processing because some regulation of pytorch    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)    
    
    #deliver the image to the processor and pass through the model
    image = image.to(DEVICE)
    detections = model(image)[0]   

    #pick out low confidence and draw boxes
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > MIN_CONFIDENCE:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            print(startX, startY, endX, endY)
            w = endX - startX
            h = endY - startY
            object_count += 1 #for naming the new object
            subprocess.call(['ffmpeg', '-i', path_input_image, '-filter:v', f'crop={w}:{h}:{startX}:{startY}', f'output_images/object_{object_count}.png'])

    #output image
    cv2.imwrite(path_output_image, orig)

    


