import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


FLANN_INDEX_KDTREE = 0
MIN_MATCH_COUNT = 10

# reference image
canvas = cv2.imread('./reference.jpg')

# camera settings
brightness = 170
contrast = 100
focusvalue = 66

cap = cv2.VideoCapture(0)
cap.set(28, focusvalue)
cap.set(10 , brightness ) # brightness     min: 0   , max: 255 , increment:1  
cap.set(11, contrast   ) 

# Pre-process the piece
def identify_contour(piece, threshold_low=160, threshold_high=255):
    piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(piece,(5,5),0) 
    ret, thresh = cv2.threshold(blur, threshold_low, threshold_high, 0) 
    cv2.imshow("thresh",thresh) 
    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sorted = np.argsort(map(cv2.contourArea, contours))
    return contours, contour_sorted[-2]

def get_bounding_rect(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h


 # Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()
img2 = canvas.copy() # copy referenceimage 

# detect artifacts
kp2, des2 = sift.detectAndCompute(img2,None)


while True:
  
    isValid = False
    ret, piece = cap.read()

    try:
         # Get the contours
        contours, contour_index = identify_contour(piece.copy())

        # Get a bounding box around the piece
        x, y, w, h = get_bounding_rect(contours[contour_index])
        
        cropped_piece = piece.copy()[y:y+h, x:x+w]
        isValid = True
    except:
        isValid = False
   

    
    print(isValid)
    if isValid:

        img1 = cropped_piece.copy() 
        cv2.imshow("piece",img1)
        
        # find piece artifacts
        kp1, des1  = sift.detectAndCompute(img1,None)


        
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            d,h,w = img1.shape[::-1]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        scale_percent = 40 # percent of original size
        width = int(img3.shape[1] * scale_percent / 100)
        height = int(img3.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow("result",resized)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1000) & 0xFF == ord('r'): # reset reference
        img2 = canvas.copy() 

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

