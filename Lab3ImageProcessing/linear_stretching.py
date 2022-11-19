import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def linear_stretching(img,pmin,pmax):
    out = 255.*(img-pmin)/(pmax-pmin)
    return out
def percentile_linear_stretching(img,pmin,pmax):
    img[img<pmin]=pmin
    img[img>pmax]=pmax
    out = 255.*(img-pmin)/(pmax-pmin)
    return out

def plot_img_hist(stretched_img,hist_stretch,title):
    plt.subplot(1,2,1)
    plt.title('Original img')
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.subplot(1,2,2)
    plt.title(title)
    plt.imshow(stretched_img,cmap='gray',vmin=0,vmax=255)
    plt.show()

    plt.subplot(1,2,1)
    plt.title('Original img')
    plt.stem(hist)
    plt.subplot(1,2,2)
    plt.title(title)
    plt.stem(hist_stretch)
    plt.show()

def calculate_percentile(hist,percentile):
    tot_pixels = np.sum(hist)
    target=tot_pixels*percentile/100
    count=0
    for i in range(np.size(hist)):
        if(count>=target): #se ho raggiunto il target ritorno il valore di quel livello
            return i
        else: 
            count+=hist[i] #incremento con numero di pixel associati a quel livello


#Reading the img
img = cv.imread("ex/image.png", cv.IMREAD_GRAYSCALE)
hist, bins = np.histogram(img.flatten(), 256, [0,256])
print(calculate_percentile(hist,5))
print(calculate_percentile(hist,95))


#Pmin,Pmax = The minimum and maximum value of the image respectively. 
stretched_img = linear_stretching(np.copy(img),np.min(img.flatten()),np.max(img.flatten()))
hist_stretch, bins_stretch = np.histogram(stretched_img.flatten(), 256, [0,256])
plot_img_hist(stretched_img,hist_stretch,"Pmin and Pmax as min and max value")


#Pmax=40 , Pmin = 0
stretched_img = linear_stretching(np.copy(img),0,40)
hist_stretch, bins_stretch = np.histogram(stretched_img.flatten(), 256, [0,256])
plot_img_hist(stretched_img,hist_stretch,"Pmin=0 Pmax=40")

#pmin=5% and Pmax=95%
stretched_img = percentile_linear_stretching(np.copy(img),calculate_percentile(hist,5),calculate_percentile(hist,95))
hist_stretch, bins_stretch = np.histogram(stretched_img.flatten(), 256, [0,256])
plot_img_hist(stretched_img,hist_stretch,"pmin=5% and Pmax=95%")





