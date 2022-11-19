import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def percentile_linear_stretching(img,pmin,pmax):
    img[img<pmin]=pmin
    img[img>pmax]=pmax
    out = 255.*(img-pmin)/(pmax-pmin)
    return out

def calculate_percentile(hist,percentile):
    tot_pixels = np.sum(hist)
    target=tot_pixels*percentile/100
    count=0
    for i in range(np.size(hist)):
        if(count>=target): #se ho raggiunto il target ritorno il valore di quel livello
            return i
        else: 
            count+=hist[i] #incremento con numero di pixel associati a quel livello


def play_video_modified_frame(video_path):
    cap = cv2.VideoCapture('ex/1.avi')
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        #gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #print(gray_frame.shape)
        #frame = frame.astype(float) #!!
        hist, bins = np.histogram(frame.flatten(), 256, [0,256])
        vmin = calculate_percentile(hist,5)
        vmax = calculate_percentile(hist,95)
        #new_frame = percentile_linear_stretching(frame,vmin,vmax)
        new_frame_gauss = cv2.GaussianBlur(frame,(9,9),1.5)
        new_frame_bil = cv2.bilateralFilter(frame,9,75,75)
        #new_frame = new_frame.astype(np.uint8) #!!
        # Displaying with OpenCV
        cv2.imshow('frame', new_frame_gauss)
        cv2.imshow('frame1',new_frame_bil)
        # Stop playing when entered 'q' from keyboard
        if cv2.waitKey(25) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="LabSession3Images/video.avi", help="path to video")
    args = parser.parse_args()
    play_video_modified_frame(args.video_path)
