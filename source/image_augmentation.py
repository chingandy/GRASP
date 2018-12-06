import cv2
import numpy as np



#Rotate image by angle
def rotate_img(img, angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst[0,:] = 255
    dst[:,0] = 255
    return dst

#Flip image vertically
def flip_img(img):

    vertical_img = cv2.flip(img, 1)
    return vertical_img

#Move picture one pixel to the right
def translate_right(img):
    rows,cols = img.shape
    M = np.float32([[1,0,1],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,0] = 255
    return translated_img

#Move picture one pixel to the left
def translate_left(img):
    rows,cols = img.shape
    M = np.float32([[1,0,-1],[0,1,0]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[:,-1] = 255
    return translated_img

#Move picture one pixel up
def translate_up(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,-1]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[-1,:] = 255
    return translated_img

#Move picture one pixel down
def translate_down(img):
    rows,cols = img.shape
    M = np.float32([[1,0,0],[0,1,1]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))
    translated_img[0,:] = 255
    return translated_img


def augment_img(img, translate, flip, rot):
    #Do translations on the image
    if translate == 0:
        aug_img = img
    elif translate == 1:
        aug_img = translate_right(img)
    elif translate == 2:
        aug_img = translate_left(img)
    elif translate == 3:
        aug_img = translate_up(img)
    elif translate == 4:
        aug_img = translate_down(img)
    elif translate == 5:
        aug_img = translate_right(img)
        aug_img = translate_up(aug_img)
    elif translate == 6:
        aug_img = translate_right(img)
        aug_img = translate_down(aug_img)
    elif translate == 7:
        aug_img = translate_left(img)
        aug_img = translate_up(aug_img)
    elif translate == 8:
        aug_img = translate_left(img)
        aug_img = translate_down(aug_img)
    else:
        print("Not valid argument for translation")

    #Flip the image
    if flip == 0:
        pass
    elif flip == 1:
        aug_img = flip_img(aug_img)
    else:
        print("Not valid argument for flipping")

    #Rotate the image
    if rot == 0:
        pass
    elif rot == 1:
        aug_img = rotate_img(aug_img, 90)
    elif rot == 2:
        aug_img = rotate_img(aug_img, 180)
    elif rot == 3:
        aug_img = rotate_img(aug_img, 270)
    else:
        print("Not valid argument for rotation")

    return aug_img
