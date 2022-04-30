import cv2, os
haar_file = 'haarcascade_frontalface_default.xml' #You can find this file in official OpenCV Github repo "https://github.com/opencv/opencv/tree/master/data/haarcascades" and save the file in the folder where you save this program. Or i have given the file along with this program, you can download that. 
datasets = 'dataset' # Main folder name where you save different individual faces
sub_data = 'Your_Name' # Folder name where you save single person face images    

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100) # Standard resize values for better performance, you can change this if yyou want.
face_cascade = cv2.CascadeClassifier(haar_file) # Load the haarcascade file.
webcam = cv2.VideoCapture(0) # If you have multiple cameras give proper camera id in place of 0

count = 1
while count < 31: # Saves 30 images
    #print(count) 
    (_, im) = webcam.read() # Read the frame from camera
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # Convert BGR image to Gray scale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) # Detect face/faces in the gray scale image from haarcascade file
    for (x,y,w,h) in faces: # Loop through all the faces detected
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2) # Put rectangle box around the faces
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height)) # Rsize to standard width and height.
        cv2.imwrite('%s/%s.png' % (path,count), face_resize) # Save the face image in the folder. 
    count += 1
	
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27: # Press 'q' or 'esc' to stop the execution
        break
print("Dataset obtained successfully")
webcam.release() # Release the webcam
cv2.destroyAllWindows() # Destroy the window showing images
