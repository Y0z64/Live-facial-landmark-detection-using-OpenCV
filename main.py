import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

#Parameters----------------------------------------------------------------------------
#Face detection
face_range = 0 #0 for face wihtin 0-2m from camera | 1 for face within 2-5m from camera
confidence = 0.75 #Percentage to take a prediction as succesfull | %50 by default | range 0.0 to 1.0

#Image
#img_name = "two_faces_close.jpg"
#img_path = 'samples_groups/' + img_name
max_face_num = 1

#Video
#video_res =
#video_aspect = 
max_face_num_v = 1

#Initialization of tools---------------------------------------------------------------
#initialize face mediapipe face detection
mp_face_detection = mp.solutions.face_detection

#Setup face selection function
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

# Initialize mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils

#Initialization for mesh specific processes
#Initialize the mediapipe facemesh
mp_face_mesh = mp.solutions.face_mesh

#Setup the face landmarks function for images.
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=max_face_num,
                                         min_detection_confidence = confidence)

#Setup the face landmarks fucntion for videos
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_face_num_v,
                                         min_detection_confidence = confidence)

#Initialize the drawing styles
mp_drawing_styles = mp.solutions.drawing_styles

#Initialization for live video feed
#Initilize video in 4:3 (1280,960 res)
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1792)
camera_video.set(4,1344)

#Read and display image---------------------------------------------------------------
#Read an image from specified path
#sample_img = cv2.imread(img_path)

#Detect landmarks and generate face mesh-------------------------------------------
def detectFacialLandmarks(image, face_mesh, display = False):

    #detect facial landmarks
    results = face_mesh.process(image[:,:,::-1])

  #image copy translated to RGB
    output_image = image[:,:,::-1].copy()

    #Check if facial landmarks are found
    if results.multi_face_landmarks:
        #Iterate over the found faces
        for face_landmarks in results.multi_face_landmarks:
            #Draw the landmarks in the mesh_copy with the face mesh tesselation connections using default face mesh tesselation style
            mp_drawing.draw_landmarks(image=output_image, 
                                    landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None, 
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=output_image,
                                    landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())


    #Check if the output and input image are specified to be displayed
    if display:
        #Display the original image and the output image
        plt.figure(figsize=[15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image);plt.title("Output");plt.axis('off');
        plt.show()
    else:
        return np.ascontiguousarray(output_image[:,:,::-1],dtype=np.uint8), results


#Create a window
#cv2.namedWindow("Face recognition",cv2.WINDOW_NORMAL)

#Initialize time
time1 = 0

while camera_video.isOpened():
    #Read frame
    ok, frame = camera_video.read()

    #check if frame is read properly if false skip frame | This will limit your life feed fps to the program output
    if not ok:
        continue

    #Flip the frame horizontally | there is more options to flip 1 = Selfie view?
    frame = cv2.flip(frame,1)

    #Perform detection
    frame, _ = detectFacialLandmarks(frame, face_mesh_videos, display=False)

    #Set the time for this frame to be the current time
    time2 = time()
    
    #Check diff between the previows frame and this
    if (time2 - time1) > 0:
        #Calculate fps
        fps = 1.0/(time2-time1)
        #Write fps on display
        cv2.putText(frame, "FPS {}".format(int(fps)),(10,30),
                    cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    
    #Ipdate previous time fraeme tho this one
    time1 = time2

    #Display frame
    cv2.imshow("Face landmaks detection", frame)
    
    #Listen for ASCII keys every 1ms | If its ESC key close program
    k = cv2.waitKey(1) & 0xFF
    if(k==27):
        break
    if(k==66 or 98):
        cv2.putText(frame, "balls.", (20,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)

#close windows and video
camera_video.release()
cv2.destroyAllWindows()
        