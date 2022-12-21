import cv2
import numpy as np
import itertools
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

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


def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
    '''
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
        face_part:      The name of the face part on which the filter image will be overlayed.
        INDEXES:        The indexes of landmarks of the face part.
        display:        A boolean value that is if set to true the function displays 
                        the annotated image and returns nothing.'''
    
    #set RGB if necesary
    annotated_img = image.copy()

    #Error handling for resize of the image
    try:
        #get x,y of image
        filter_img_height, filter_img_width,_ = filter_img.shape()
        _,face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)
        required_heigth = int(face_part_height*2.5)
        #keep aspect ratio
        resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*(required_heigth/filter_img_height)),required_heigth))
        #get new sizes
        filter_img_height,filter_img_width,_ = resized_filter_img.shape
        #convert to B&W to get the mask image
        _,filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY)
                                                       ,25,255,cv2.THRESH_BINARY_INV)
        #Calculate the center of the face part
        center = landmarks.mean(axis=0).astype("int")

        #Check if the face part is true | Customize this with your own face parts
        if face_part == "MOUTH":
            #Calculate the location where the smoke filter will be placed
            location = (int(center[0]-filter_img_width / 3), int(center[1]))
        else: #if an eye
            location = (int(center[0]-filter_img_width/2),int(center[1]-filter_img_height/2))
        
        #retrieve the region from the image where the filter will be placed
        ROI = image[location[1]: location[1]+filter_img_height,
                    location[0]: location[0]+filter_img_width]
        #Set the pixels in the region to 0 using Bitwise-AND
        resultant_img = cv2.bitwise_and(ROI,ROI, mask=filter_img_mask)
        #Add the resultant image and the resized filter image by updating the pixes set to 0 to those set by the img
        resultant_img = cv2.add(resultant_img, resized_filter_img)

        #Update the region with the resultant image
        annotated_img[location[1]: location[1]+filter_img_height,
                      location[0]: location[0]+filter_img_width] = resultant_img
    
    except Exception as e:
        pass
    
    #Check if its specified to be displayed
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(annotated_img[:,:,::-1]);plt.title("Face filter applied");plt.axis('off');
    else:
        return annotated_img

# #Function to overlay
# def overlay(image, filter_img, face_landmarks, face_part, INDEXES, display=True):
#     '''
#     This function will overlay a filter image over a face part of a person in the image/frame.
#     Args:
#         image:          The image of a person on which the filter image will be overlayed.
#         filter_img:     The filter image that is needed to be overlayed on the image of the person.
#         face_landmarks: The facial landmarks of the person in the image.
#         face_part:      The name of the face part on which the filter image will be overlayed.
#         INDEXES:        The indexes of landmarks of the face part.
#         display:        A boolean value that is if set to true the function displays 
#                         the annotated image and returns nothing.
#     Returns:
#         annotated_image: The image with the overlayed filter on the top of the specified face part.
#     '''
    
#     # Create a copy of the image to overlay filter image on.
#     annotated_image = image.copy()
    
#     # Errors can come when it resizes the filter image to a too small or a too large size .
#     # So use a try block to avoid application crashing.
#     try:
    
#         # Get the width and height of filter image.
#         filter_img_height, filter_img_width, _  = filter_img.shape

#         # Get the height of the face part on which we will overlay the filter image.
#         _, face_part_height, landmarks = getSize(image, face_landmarks, INDEXES)
        
#         # Specify the height to which the filter image is required to be resized.
#         required_height = int(face_part_height*2.5)
        
#         # Resize the filter image to the required height, while keeping the aspect ratio constant. 
#         resized_filter_img = cv2.resize(filter_img, (int(filter_img_width*
#                                                          (required_height/filter_img_height)),
#                                                      required_height))
        
#         # Get the new width and height of filter image.
#         filter_img_height, filter_img_width, _  = resized_filter_img.shape

#         # Convert the image to grayscale and apply the threshold to get the mask image.
#         _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
#                                            25, 255, cv2.THRESH_BINARY_INV)

#         # Calculate the center of the face part.
#         center = landmarks.mean(axis=0).astype("int")

#         # Check if the face part is mouth.
#         if face_part == 'MOUTH':

#             # Calculate the location where the smoke filter will be placed.  
#             location = (int(center[0] - filter_img_width / 3), int(center[1]))

#         # Otherwise if the face part is an eye.
#         else:

#             # Calculate the location where the eye filter image will be placed.  
#             location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))

#         # Retrieve the region of interest from the image where the filter image will be placed.
#         ROI = image[location[1]: location[1] + filter_img_height,
#                     location[0]: location[0] + filter_img_width]

#         # Perform Bitwise-AND operation. This will set the pixel values of the region where,
#         # filter image will be placed to zero.
#         resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)

#         # Add the resultant image and the resized filter image.
#         # This will update the pixel values of the resultant image at the indexes where 
#         # pixel values are zero, to the pixel values of the filter image.
#         resultant_image = cv2.add(resultant_image, resized_filter_img)

#         # Update the image's region of interest with resultant image.
#         annotated_image[location[1]: location[1] + filter_img_height,
#                         location[0]: location[0] + filter_img_width] = resultant_image
            
#     # Catch and handle the error(s).
#     except Exception as e:
#         pass
    
#     # Check if the annotated image is specified to be displayed.
#     if display:

#         # Display the annotated image.
#         plt.figure(figsize=[10,10])
#         plt.imshow(annotated_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
#     # Otherwise
#     else:
            
#         # Return the annotated image.
#         return annotated_image


#MAIN---------------------------------------------------------------------------------------------
#Initilize video in 4:3 (1280,960 res)
camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)
 
# Create named window for resizing purposes.
cv2.namedWindow('Face Filter', cv2.WINDOW_NORMAL)

# Read the left and right eyes images.
left_eye = cv2.imread('filters/googly_eye.png')
right_eye = cv2.imread('filters/black_rectangle.png')
 
# Create blank image
blank_image = np.zeros((left_eye.shape[0], left_eye.shape[1], 3), dtype=np.uint8)


# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue

    #flip the frame
    frame = cv2.flip(frame, 1)

    #set the blank rectangle in frame
    cv2.rectangle(blank_image, (0,0), (left_eye.shape[1], left_eye.shape[0]), (0,0,0), -1)

    # Perform Face landmarks detection.
    _,face_mesh_results= detectFacialLandmarks(frame, face_mesh_videos, display=False)

    if face_mesh_results.multi_face_landmarks:
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            frame = overlay(frame, left_eye[:,:,::-1], face_landmarks,
                'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False,)
            frame = overlay(frame, left_eye, face_landmarks,
                            'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False,)

    # Display the frame.
    cv2.imshow('Face Filter', frame)

    # Check if 'ESC' is pressed and break the loop.
    k = cv2.waitKey(1) &  0xFF
    if(k == 27):
        break
 
# Release the VideoCapture Object and close the windows.                  
camera_video.release()
cv2.destroyAllWindows()