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
img_name = "2.jpg"
img_path = 'samples/' + img_name
max_face_num = 2


#Initialization of tools---------------------------------------------------------------
#initialize face mediapipe face detection
mp_face_detection = mp.solutions.face_detection

#Setup face selection function
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

# Initialize mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils

#Initialize the mediapipe facemesh
mp_face_mesh = mp.solutions.face_mesh

#Setup the face landmarks function for images.
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=max_face_num,
                                         min_detection_confidence = confidence)

#Initialize the drawing styles
mp_drawing_styles = mp.solutions.drawing_styles


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
                                                                               circle_radius=10))

#Detect landmarks and generate face mesh-------------------------------------------
#Perfrom landmarks detection
face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])

#Get list of indexes in given face part | You can get the indexes for other face parts from mediapipe (Insert link/list here)
face_parts = dict({
    "LEFT_EYE":list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE))),
    "RIGHT_EYE":list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
})

#Check for found facial landmarks
if face_mesh_results.multi_face_landmarks:

    #Iterate over the found faces
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        print(f"Face number: {face_no+1}")
        print("----------------------------")

        #Display the landmarks of given face part | This code uses your variable name to print said landmark, will remove later
        for face_part, part_indexes in face_parts.items():
            print(f'{face_part}_LANDMARKS:\n')

            for part_index in part_indexes[:2]:
                #Display the found normalized indexes in given face part
                print(face_landmarks.landmark[part_index])

#image copy translated to RGB
mesh_copy = sample_img[:,:,::-1].copy()

#Check if facial landmarks are found
if face_mesh_results.multi_face_landmarks:
    #Iterate over the found faces
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        #Draw the landmarks in the mesh_copy with the face mesh tesselation connections using default face mesh tesselation style
        mp_drawing.draw_landmarks(image=mesh_copy, 
                                  landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        mp_drawing.draw_landmarks(image=mesh_copy,
                                  landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())              


#Display sample image, result and mesh side by side by using sub plots---------------------------------------------------
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,3,1)
plt.title("Sample image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);
ax1 = fig.add_subplot(1,3,2)
plt.title("Result");plt.axis("off");plt.imshow(img_copy);
ax2 = fig.add_subplot(1,3,3)
plt.title("Mesh image");plt.axis("off");plt.imshow(mesh_copy);
plt.show()
