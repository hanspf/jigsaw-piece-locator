import cv2
import numpy as np
from matplotlib import pyplot as plt



brightness = 170
contrast = 100
focusvalue = 66


cap = cv2.VideoCapture(0)

cap.set(28, focusvalue)
#cap.set(3,1280)
#cap.set(4,1024)
cap.set(10 , brightness ) # brightness     min: 0   , max: 255 , increment:1  
cap.set(11, contrast   ) 

while True:

    ret, piece = cap.read()
    #cv2.imwrite('./focus.jpg', piece)

    bwimage = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', bwimage)
    #bwimage = cv2.equalizeHist(bwimage)
    #cv2.imwrite('./bwimage.jpg', bwimage)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(bwimage,(5,5),0)
    cv2.imshow('Video', blur)
    ret3,blurredimage = cv2.threshold(blur,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #cv2.imwrite('./threshold.jpg', blurredimage)

    # Display the resulting frame
    cv2.imshow('Video', blurredimage)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

