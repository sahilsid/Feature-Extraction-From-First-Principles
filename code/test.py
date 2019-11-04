from preprocess import *
import matplotlib.pyplot as plt
from filter import *
import math
from canny import *
from harris import *
from scipy import ndimage
from scipy.ndimage.filters import convolve
import os
import numpy as np
from PIL import Image

files= os.listdir("../data")
images=[]

for image in files:
    images.append(load_image("../data/"+str(image)))
    
    
for i,img in enumerate(images,0):
    new = np.asarray(img[:,:,0])
    c = Canny(new,filename=files[i])
    h = Harris(new,filename=files[i])



def generate_final_output():

    grad_mag=[]
    original=[]
    grad_dir=[]
    thinned=[]
    thresholded=[]
    linked=[]
    for i,image in enumerate(files,0):
        filename = image.rpartition(".")[0]+'.png'
        original.append( plt.imread("../results/canny/original/"+str(filename)))
        grad_mag.append( plt.imread("../results/canny/grad_mag/"+str(filename)))
        grad_dir.append( plt.imread("../results/canny/grad_dir/"+str(filename)))
        thinned.append( plt.imread("../results/canny/thinned/"+str(filename)))
        thresholded.append( plt.imread("../results/canny/thresholded/"+str(filename)))
        linked.append( plt.imread("../results/canny/linked/"+str(filename)))

        x=original[i].shape[0]
        y=original[i].shape[1]
        test = np.zeros((6*x,y,4))
        test[:1*x,:1*y]=original[i]
        test[1*x:2*x,:]=grad_mag[i]
        test[2*x:3*x,:]=grad_dir[i]
        test[3*x:4*x,:]=thinned[i]
        test[4*x:5*x,:]=thresholded[i]
        test[5*x:6*x,:]=linked[i]

        plt.figure(i+1,figsize=(6*x,6*y))
        plt.axis("off")
        fig=plt.imshow(test,interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig("../results/combined/"+str(filename))
        plt.show()

# generate_final_output()