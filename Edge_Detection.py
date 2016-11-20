from cv2 import *
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import math

def zero_cross(img):
    zero_img=np.zeros(img.shape)
    for i in range(len(img)):
        for j in range(len(img[0])-1):
            if(img[i][j]>0 and img[i][j+1]<0):
                zero_img[i][j]=255;
            if(img[i][j]<0 and img[i][j+1]>0):
                zero_img[i][j+1]=255;
    for i in range(len(img[0])):
        for j in range(len(img)-1):
            if(img[j][i]>0 and img[j+1][i]<0):
                zero_img[j][i]=255;
            if(img[j][i]<0 and img[j+1][i]>0):
                zero_img[j+1][i]=255;
            
    return zero_img



def mag(img1,img2):
    img=np.zeros(img1.shape)
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            x=math.sqrt((img1[i][j]**2+img2[i][j]**2))
            if(x<=255):
                img[i][j]=x
            else:
                img[i][j]=255
    return img
            

def comp(img1,img2):
    img=np.zeros(img1.shape)
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            if(img1[i][j]==img2[i][j] and img1[i][j]==255):
                img[i][j]=255
    return img

img = imread("UBCampus.jpg",0)

kernel=np.asarray([
        [0,0,-1,-1,-1,0,0],
        [0,-2,-3,-3,-3,-2,0],
        [-1,-3,5,5,5,-3,-1],
        [-1,-3,5,16,5,-3,-1],
        [-1,-3,5,5,5,-3,-1],
        [0,-2,-3,-3,-3,-2,0],
        [0,0,-1,-1,-1,0,0]])

log_kernel=np.asarray([
        [0,0,1,0,0],
        [0,1,2,1,0],
        [1,2,-16,2,1],
        [0,1,2,1,0],
        [0,0,1,0,0]])

img=GaussianBlur(img,(3,3),0)
dog=signal.convolve2d(img, kernel, boundary='symm', mode='same')
zero_img_dog=zero_cross(dog)
log=signal.convolve2d(img, log_kernel, boundary='symm', mode='same')
zero_img_log=zero_cross(log)



sobelx = Sobel(img,CV_64F,1,0,ksize=3)
sobely = Sobel(img,CV_64F,0,1,ksize=3)

sobel=mag(sobelx,sobely)

sobel=sobel.astype("uint8")


sobel=threshold(sobel,128,255,THRESH_BINARY)[1]


final_dog=comp(sobel,zero_img_dog)
final_log=comp(sobel,zero_img_log)


plt.subplot(211),plt.imshow(dog, cmap = 'gray')
plt.title("Difference of Gaussian")
plt.subplot(212),plt.imshow(log, cmap = 'gray')
plt.title("Lapacian of Gaussian")


plt.figure(2)
plt.subplot(211),plt.imshow(zero_img_dog, cmap = 'gray')
plt.title("Zero-Crossing-DOG")
plt.subplot(212),plt.imshow(zero_img_log, cmap = 'gray')
plt.title("Zero-Crossing-LOG")


plt.figure(3)
plt.subplot(211),plt.imshow(final_dog, cmap = 'gray')
plt.title("Final-DOG")
plt.subplot(212),plt.imshow(final_log, cmap = 'gray')
plt.title("Final-LOG")



imshow("Zero-Crossing-DOG",zero_img_dog)
imshow("Zero-Crossing-LOG",zero_img_log)

imshow("Sobel Magnitude",sobel)

imshow("Final-DOG",final_dog)
imshow("Final-LOG",final_log)



plt.show()

waitKey(0)
destroyAllWindows() 
