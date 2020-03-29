import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Import our game board
canvas = cv2.imread('./reference.jpg')
# Import our piece (we are going to use a clump for now)

brightness = 170
contrast = 100
focusvalue = 66


cap = cv2.VideoCapture(0)

cap.set(28, focusvalue)
#cap.set(3,1280)
#cap.set(4,1024)
cap.set(10 , brightness ) # brightness     min: 0   , max: 255 , increment:1  
cap.set(11, contrast   ) 

#ret, piece = cap.read()
#cv2.imwrite('./focus.jpg', piece)

#bwimage = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('./bwimage.jpg', bwimage)

# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(bwimage,(5,5),0)
#ret3,blurredimage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imwrite('./threshold.jpg', blurredimage)

#piece = cv2.imread('./test6.jpg')

# Pre-process the piece
def identify_contour(piece, threshold_low=150, threshold_high=255):
    """Identify the contour around the piece"""
    piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(piece,(5,5),0)

    ret, thresh = cv2.threshold(piece, threshold_low, threshold_high, 0)
    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_sorted = np.argsort(map(cv2.contourArea, contours))
    return contours, contour_sorted[-2]

def get_bounding_rect(contour):
    """Return the bounding rectangle given a contour"""
    x,y,w,h = cv2.boundingRect(contour)
    return x, y, w, h


img2 = canvas.copy() # trainImage

#cv2.imwrite('./piece.jpg', img1)

# cv2.imshow("1",img1)
# cv2.imshow("2",img2)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
# cv2.waitKey(1)

 # Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()
print("detect on source image")
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

        
        img1 = cropped_piece.copy() # queryImage
        cv2.imshow("piece",img1)
        #try:
        #   f = open("keypoints.data", "rb")
        #  datafile = f.read()
        #  print("datafile found")
        #  f.close()
        #except IOError:
        #    # find the keypoints and descriptors with SIFT
        datafile = sift.detectAndCompute(img1,None)
            #with open("keypoints.data", "wb") as binary_file:
                # Write text or bytes to the file
                #binary_file.write(bytearray(str(datafile)))
            
        kp1, des1 = datafile

        

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
        #     good.append(m)
            if m.distance < 0.9*n.distance:
                good.append(m)


        MIN_MATCH_COUNT = 10

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
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        #cv2.imwrite('./solution.jpg', img3)

        scale_percent = 30 # percent of original size
        width = int(img3.shape[1] * scale_percent / 100)
        height = int(img3.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow("result",resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

