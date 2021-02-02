#!/usr/bin/env python
import cv2
import numpy as np
import time
n1=240
n2=320
n3=n1*n2
hmin=0
hmax=47
smin=138
smax=216
vmin=60
vmax=142
#lower_blue = np.array([110,50,50])
#upper_blue = np.array([130,255,255])


def talker():
    time.sleep(4)
    seuil=10
    kernel=np.ones((5,5),np.uint8)
    cap=cv2.VideoCapture(0)
    cap.set(3,n2)
    cap.set(4,n1)
    cv2.namedWindow('closing')
    cv2.namedWindow('camera')

    while not False:
        X=[]
        Y=[]

        _, frame = cap.read()
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hue,sat,val=cv2.split(hsv)

        hthresh=cv2.inRange(np.array(hue),hmin,hmax)
        sthresh=cv2.inRange(np.array(sat),smin,smax)
        vthresh=cv2.inRange(np.array(val),vmin,vmax)
        tracking=cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))
        dilation=cv2.dilate(tracking,kernel,iterations=3)
        erosion=cv2.erode(dilation,kernel,iterations=5)
        opening=cv2.morphologyEx(erosion,cv2.MORPH_OPEN,kernel)
        closing=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)

        cv2.imshow('closing',closing)
        cv2.imshow('camera',frame)

        pixel=cv2.countNonZero(closing)
        pourcpixel=pixel*1000/n3

        #msg=herbe()
        
        if pourcpixel>seuil:

            #msg.herbe=True            
            #print("Objet orange trouvé")

            ret,thresh=cv2.threshold(closing,127,255,0)
            contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(cnts) > 0:
                # On veut que le plus grand contour
                contour = max(cnts, key=cv2.contourArea)
                M = cv2.moments(contour)
                contour_area = M['m00']
                if contour_area > 0:
                    cX = int(M['m10']/M['m00'])
                    cY = int(M['m01']/M['m00'])

                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                (xg, yg, wg, hg) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
                cv2.putText(frame, "Objet orange", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Image", frame)
                distance = np.sqrt(1160000/contour_area)
                #print("area = " + str(contour_area))
                print("distance = " + '{:.4}'.format(distance) + " cm")
            
            #msg.X=X
            #print("X = "+str(X))
            #print("Y = "+str(Y))

            #msg.Y=Y
            #pub.publish(msg)
                
        if pourcpixel<=seuil:
            #msg.herbe=False
            #pub.publish(msg)
            #print("False")
            pass

        k=cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    

    


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
