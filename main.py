import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

#Parameters----------------------------------------------------------------------------
#Face detection
face_range = 0 #0 for face wihtin 0-2m from camera | 1 for face within 2-5m from camera
confidence = 0.75 #Percentage to take a prediction as succesfull | %50 by default | range 0.0 to 1.0
spec_color = (255,0,0)

#Image
img_name = "myface.jpeg"
img_path = 'samples/' + img_name


#Initialization of tools---------------------------------------------------------------
#initialize face mediapipe face detection
mp_face_detection = mp.solutions.face_detection

#Setup face selection function
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

# Initialize mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils


#Read and display image---------------------------------------------------------------
#Read an image from specified path
sample_img = cv2.imread(img_path)


#Face detection------------------------------------------------------------------------
#face detection results of the converted to format RGB image
face_detection_results = face_detection.process(sample_img[:,:,::-1])

#check if the faces in the image are found.
if face_detection_results.detections:
    #Iterate over the found faces
    for face_no,face in enumerate(face_detection_results.detections):

        #Display the face number upon which we are iterating upon
        print(f"Face num: {face_no+1}")
        print("------------------------------")

        #Display face confidence | note: the faces below the specified face confidence will be ignored
        print(f"Face confidence: {round(face.score[0], 2)}")

        #display the face bounding box and key points coordinates
        face_data = face.location_data

        #Display the face bounding box coordinates
        print(f"\n Face box:\n{face_data.relative_bounding_box}")
            #this will print the data with the x_max and width, they are both normalized by the image proportions

        #Iterate two times as we only want to display the two first key points of each detected face | 6 in total
        for i in range(2):
            #display the found normalized key points
            print(f"{mp_face_detection.FaceKeyPoint(i).name}")
            print(f"{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}")


#Display results------------------------------------------------------------------------
#display the results on a copy of the image using mp
img_copy = sample_img[:,:,::-1].copy()

#Check if the faces are found
if face_detection_results.detections:
    #iterate over the found faces
    for face_no,face in enumerate(face_detection_results.detections):
        #draw the box and key points
        mp_drawing.draw_detection(image=img_copy, detection=face,
                                  keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0),
                                                                               thickness=2,
                                                                               circle_radius=2))

#Display sample image and result side by side by using sub plots
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,2,1)
plt.title("Sample image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);
ax1 = fig.add_subplot(1,2,2)
plt.title("Result");plt.axis("off");plt.imshow(img_copy);
plt.show()