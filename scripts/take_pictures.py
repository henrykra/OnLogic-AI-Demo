import cv2 as cv
import os
import re
import time

# Script for taking photos with the webcam 


def take_pictures(images_path, max_images, cam=0, auto=True):
    """Function for taking pictures
    
    Args:
        cam: integer representing which webcam to use. 
        auto: bool for taking pictures every half second or on pressing 'y'"""
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    
    # initialize webcam feed
    cap = cv.VideoCapture(cam)

    # find how many images are already present in the folder of the same format
    try:
        startnum = max([int(re.search(r'\d+', file).group()) for file in os.listdir(images_path)]) + 1
    # startnum is one more than the largest numbered image file in the folder
    except ValueError:
        startnum = 0

    c = startnum

    while(True):
        ret, frame = cap.read()
        if ret:
            cv.imshow('image', frame)
        else:
            print('Capture Failed')
            break

        if auto:
            time.sleep(.5)
            cv.imwrite(os.path.join(images_path, f'img{c}.png'), frame)
            c += 1
            print(f'{c} picture(s) saved at {images_path}')


        else:
            if cv.waitKey(1) & 0xFF == ord('y'):
                cv.imwrite(os.path.join(images_path, f'_img{c}.png'), frame)
                c += 1
                print(f'{c} picture(s) saved at {images_path}')

        if c - startnum >= max_images:
            break
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    images_path = 'test_images'
    take_pictures(images_path, max_images=1, cam=0, auto=True)

