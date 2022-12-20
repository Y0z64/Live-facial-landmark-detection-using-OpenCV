import cv2
import numpy as np
import itertools
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

#Function to get size of the face
def getSize(image, face_landmarks, INDEXES):
    '''
    This function calculate the height and width of a face part utilizing its landmarks.
    Args:
        image:          The image of person(s) whose face part size is to be calculated.
        face_landmarks: The detected face landmarks of the person whose face part size is to 
                        be calculated.
        INDEXES:        The indexes of the face part landmarks, whose size is to be calculated.
    Returns:
        width:     The calculated width of the face part of the face whose landmarks were passed.
        height:    The calculated height of the face part of the face whose landmarks were passed.
        landmarks: An array of landmarks of the face part whose size is calculated.
    '''
    
    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Convert the indexes of the landmarks of the face part into a list.
    INDEXES_LIST = list(itertools.chain(*INDEXES))
    
    # Initialize a list to store the landmarks of the face part.
    landmarks = []
    
    # Iterate over the indexes of the landmarks of the face part. 
    for INDEX in INDEXES_LIST:
        
        # Append the landmark into the list.
        landmarks.append([int(face_landmarks.landmark[INDEX].x * image_width),
                               int(face_landmarks.landmark[INDEX].y * image_height)])
    
    # Calculate the width and height of the face part.
    _, _, width, height = cv2.boundingRect(np.array(landmarks))
    
    # Convert the list of landmarks of the face part into a numpy array.
    landmarks = np.array(landmarks)
    
    # Retrurn the calculated width height and the landmarks of the face part.
    return width, height, landmarks

#Function to detect open mouths
def isOpen(image, face_mesh_results, face_part, threshold=5, display=True):
    '''
    This function checks whether the an eye or mouth of the person(s) is open, 
    utilizing its facial landmarks.
    Args:
        image:             The image of person(s) whose an eye or mouth is to be checked.
        face_mesh_results: The output of the facial landmarks detection on the image.
        face_part:         The name of the face part that is required to check.
        threshold:         The threshold value used to check the isOpen condition.
        display:           A boolean value that is if set to true the function displays 
                           the output image and returns nothing.
    Returns:
        output_image: The image of the person with the face part is opened  or not status written.
        status:       A dictionary containing isOpen statuses of the face part of all the 
                      detected faces.  
    '''
    
    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Create a copy of the input image to write the isOpen status.
    output_image = image.copy()
    
    # Create a dictionary to store the isOpen status of the face part of all the detected faces.
    status={}
    
    # Check if the face part is mouth.
    if face_part == 'MOUTH':
        
        # Get the indexes of the mouth.
        INDEXES = mp_face_mesh.FACEMESH_LIPS
        
        # Specify the location to write the is mouth open status.
        loc = (10, image_height - image_height//40)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap. 
        increment=-30
        
    # Check if the face part is left eye.    
    elif face_part == 'LEFT EYE':
        
        # Get the indexes of the left eye.
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE
        
        # Specify the location to write the is left eye open status.
        loc = (10, 30)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap.
        increment=30
    
    # Check if the face part is right eye.    
    elif face_part == 'RIGHT EYE':
        
        # Get the indexes of the right eye.
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE 
        
        # Specify the location to write the is right eye open status.
        loc = (image_width-300, 30)
        
        # Initialize a increment that will be added to the status writing location, 
        # so that the statuses of two faces donot overlap.
        increment=30
    
    # Otherwise return nothing.
    else:
        return
    
    # Iterate over the found faces.
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
        
         # Get the height of the face part.
        _, height, _ = getSize(image, face_landmarks, INDEXES)
        
         # Get the height of the whole face.
        _, face_height, _ = getSize(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)
        
        # Check if the face part is open.
        if (height/face_height)*100 > threshold:
            
            # Set status of the face part to open.
            status[face_no] = 'OPEN'
            
            # Set color which will be used to write the status to green.
            color=(0,255,0)
        
        # Otherwise.
        else:
            # Set status of the face part to close.
            status[face_no] = 'CLOSE'
            
            # Set color which will be used to write the status to red.
            color=(0,0,255)
        
        # Write the face part isOpen status on the output image at the appropriate location.
        cv2.putText(output_image, f'FACE {face_no+1} {face_part} {status[face_no]}.', 
                    (loc[0],loc[1]+(face_no*increment)), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)
                
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
        
        # Return the output image and the isOpen statuses of the face part of each detected face.
        return output_image, status



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