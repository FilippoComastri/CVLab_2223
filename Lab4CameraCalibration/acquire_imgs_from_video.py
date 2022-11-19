import argparse
import cv2

N_IMGS = 20
FPS = 30
VIDEO_SRC = 0

def acquire_imgs_from_video(tot_imgs):
    dirname = "test_imgs/"
    img_names = [dirname + str(i) + ".jpg" for i in range(tot_imgs)]
    img_counter = 0
    frame_counter = 0
    cap = cv2.VideoCapture(VIDEO_SRC)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        cv2.imshow('frame', frame)
        frame_counter+=1
        if(frame_counter==FPS):
            frame_counter=0
            cv2.imwrite(img_names[img_counter],frame)
            print(img_names[img_counter])
            img_counter+=1
        if cv2.waitKey(25) == ord('q') or img_counter == tot_imgs:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--video_path", default="LabSession3Images/video.avi", help="path to video")
    #args = parser.parse_args()
    acquire_imgs_from_video(N_IMGS)
