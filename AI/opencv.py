import cv2 as cv
import numpy as np

videocap = cv.VideoCapture(r'C:\Users\KSHITIJ\Downloads\YTDown.com_Shorts_Objects-of-different-weights-falling-at-_Media_N4MNf3ps8uo_001_720p.mp4')
prevcircles = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2
#prevcircle is used to store the value of circle in prev frame and use it to find closest circle now
while True:
    ret, frame = videocap.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    #changing to gray scale      
    blurFrame = cv.GaussianBlur(grayFrame, (21, 21), 0)       #changing to blur image higher value in tuple more is bluring

    circle = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 
                             1.2,#resolution ratio
                               30,#dist between two circles
                             param1=100,#Edge detection threshold.
                               param2=50,  # Circle detection sensitivity
                             minRadius=20,
                               maxRadius=60)

    if circle is not None:                                      
        circle = np.uint16(np.around(circle))
        chosen = None
#i = (x, y, radius)
        for i in circle[0]:
            if chosen is None:
                chosen = i                                      
            if prevcircles is not None:
                if dist(chosen[0], chosen[1], prevcircles[0], prevcircles[1]) <= \
                   dist(i[0], i[1], prevcircles[0], prevcircles[1]):
                    chosen = i                                  

        prevcircles = chosen  # update for next frame
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)   
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 255, 255), 3)
        cv.imshow("circles", frame)
        cv.imshow("grayframe",blurFrame)

    if cv.waitKey(1) & 0xFF == ord("q"):                       
        break 

videocap.release()
cv.destroyAllWindows()