import cv2
import numpy as np

# HSV variables
h_low = 0
s_low = 50
v_low = 50
h_high = 180
s_high = 255
v_high = 255
range = 40

# Adjust webcam resolution
x_syze = 320
y_syze = 240

def HSV_video(event, x, y, flags, param):
    '''
    Function that allows the user to select the HSV values in the video
    :param x: pixel x coordinate where the mouse clicked
    :param y: pixel y coordinate where the mouse clicked
    :return: None
    '''
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsH = hsv[y, x, 0] # Gets the Hue
        colorsS = hsv[y, x, 1] # Gets the Saturation
        colorsV = hsv[y, x, 2] # Gets the Value
        colors = hsv[y, x]
        # Defines the range of values from the selected color
        colorsHMax = colorsH + Range
        colorsHMin = colorsH - Range
        colorsSMax = colorsS + Range
        colorsSMin = colorsS - Range
        colorsVMax = colorsV + Range
        colorsVMin = colorsV - Range

        print("hmin = "+str(colorsHMin))
        print("hmax = " + str(colorsHMax))
        print("smin = " + str(colorsSMin))
        print("smax = " + str(colorsSMax))
        print("vmin = " + str(colorsSMin))
        print("vmax = " + str(colorsSMax))

        cv2.setTrackbarPos('Low H', 'interface', colorsHMin)
        cv2.setTrackbarPos('Low S', 'interface', colorsSMin)
        cv2.setTrackbarPos('Low V', 'interface', colorsVMin)
        cv2.setTrackbarPos('High H', 'interface', colorsHMax)
        cv2.setTrackbarPos('High S', 'interface', colorsSMax)
        cv2.setTrackbarPos('High V', 'interface', colorsVMax)


def nothing(*args):
    pass


# Initialize the webcam
cap = cv2.VideoCapture(0)
# Sets the resolution
cap.set(3, x_syze)
cap.set(4, y_syze)

# Gets one frame of the camera and initializes the image one time out the loop to define them before calling
# the mouse callbacks, or else they would not run
ret, frame = cap.read()
cv2.namedWindow('frame')
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#image = cv2.imread("image.png")
#hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Initialize the interface
cv2.namedWindow('interface')
#cv2.setMouseCallback('interface', HSV_image)
cv2.setMouseCallback('frame', HSV_video)
cv2.createTrackbar('High H', 'interface', 0, 180, nothing)
cv2.createTrackbar('Low H', 'interface', 0, 180, nothing)
cv2.createTrackbar('High S', 'interface', 0, 255, nothing)
cv2.createTrackbar('Low S', 'interface', 0, 255, nothing)
cv2.createTrackbar('High V', 'interface', 0, 255, nothing)
cv2.createTrackbar('Low V', 'interface', 0, 255, nothing)
cv2.createTrackbar('Range', 'interface', 0, 100, nothing)
cv2.setTrackbarPos('Range', 'interface', range)

while True:
    h_low = cv2.getTrackbarPos('Low H', 'interface')
    s_low = cv2.getTrackbarPos('Low S', 'interface')
    v_low = cv2.getTrackbarPos('Low V', 'interface')
    h_high = cv2.getTrackbarPos('High H', 'interface')
    s_high = cv2.getTrackbarPos('High S', 'interface')
    v_high = cv2.getTrackbarPos('High V', 'interface')
    Range = cv2.getTrackbarPos('Range', 'interface')

    # Blocks the low values to be greater than the high values
    h_low = min(h_low, h_high)
    cv2.setTrackbarPos('Low H', 'interface', h_low)
    s_low = min(s_low, s_high)
    cv2.setTrackbarPos('Low S', 'interface', s_low)
    v_low = min(v_low, v_high)
    cv2.setTrackbarPos('Low V', 'interface', v_low)

    # Blocks the high values to be lower than the low values
    h_high = max(h_high, h_low)
    cv2.setTrackbarPos('High H', 'interface', h_high)
    s_high = max(s_high, s_low)
    cv2.setTrackbarPos('High S', 'interface', s_high)
    v_high = max(v_high, v_low)
    cv2.setTrackbarPos('High V', 'interface', v_high)

    ret, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of chosen color in HSV
    lower_range = np.array([h_low, s_low, v_low])
    upper_range = np.array([h_high, s_high, v_high])

    # Threshold the HSV image to get only selected colors
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # FindCountours
    cnts = cv2.findContours(mask.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        # Gets only the greater countour
        contour_area = max(cnts, key=cv2.contourArea)
        moment = cv2.moments(contour_area)
        if moment['m00'] > 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            # Draw its own centroid
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
        # Draw a rectangle around the contour
        (xg, yg, wg, hg) = cv2.boundingRect(contour_area)
        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.imshow('interface', image)
    cv2.resizeWindow('interface', 800, 360)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
