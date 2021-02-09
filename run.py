import numpy as np
import cv2
from time import sleep
import pafy
from bounding_box import bounding_box as bb



def center(box):
  ''' 
  box: detected bounding box as list.
  Returns: detected object center (x, y).
  '''
  x1, y1, w, h = box

  return int(x1 + (w/2)), int(y1 + (h/2))


if __name__ == '__main__':

    # Detection Settings

    # Detection min shape 80x80
    MIN_WIDTH = 80  # min width detection
    MIN_HEIGTH = 70  # min height detection

    # Detection max shape 80x80
    MAX_WIDTH = 150  # min width detection
    MAX_HEIGHT = 150  # min height detection

    DELAY = 60 # 2x fps

    LINE_POSITION_1 = 360
    LINE_POSITION_2 = 600

    # load video from YouTube
    url = 'https://www.youtube.com/watch?v=PJ5xXXcfuTc&ab_channel=Supercircuits'
    vPafy = pafy.new(url)
    play = vPafy.getbest() # get best resolution
    cap = cv2.VideoCapture(play.url) # capture Source

    # Set background subtraction
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=20, history=500)  # Pega o fundo e subtrai do que está se movendo

    while True:

        try:
            _ , frame = cap.read()  # read frame

            # some delay to the processing
            # gives output same fps as input
            tempo = float(1 / DELAY)
            sleep(tempo)  
            
            # Build the Mask
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # image to grayscale

            blured = cv2.GaussianBlur(gray, (3, 3), 5)  # blur for imperfections
            
            img_sub = backSub.apply(blured)  # apply subtraction
            
            dilated = cv2.dilate(img_sub, np.ones((5, 5)))  # increase tickness of white areas
            
            binary = np.where(dilated < 130, 0, dilated) # convert gray areas (shadows) to black
            
            # getting contours
            contours, img = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            break
        
        cv2.line(frame, (415, LINE_POSITION_1), (715, LINE_POSITION_1), (0, 255, 0), 3)        
        cv2.line(frame, (235, LINE_POSITION_2), (878, LINE_POSITION_2), (0, 0, 255), 3) 

        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x_c, y_c = center(cv2.boundingRect(c)) # getting center

            # contour validation
            if (y_c > LINE_POSITION_1) and (y_c < LINE_POSITION_2) and (x_c > 235) and (y_c < 878):
                contour_validation = (w >= MIN_WIDTH) and (h >= MIN_HEIGTH) and (w <= MAX_WIDTH) and (h <= MAX_HEIGHT)
                if contour_validation:
                    bb.add(frame, x, y, x + w, y + h, 'CAR', 'yellow') 

        cv2.imshow("Result", frame)
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()