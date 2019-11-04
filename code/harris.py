from preprocess import *
import matplotlib.pyplot as plt
from filter import *
import math
from scipy import ndimage
from scipy.ndimage.filters import convolve

class Harris():
    def __init__(self,image,threshold=0.01,filename='1'):

        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.display_images=[]
        self.display_titles=[]
        self.rgb = False
        self.window_size = 9
        self.gaussian_sigma = 2
        self.gaussian = generate_gaussian_mask(self.window_size,self.gaussian_sigma)
        self.display_images.append(image)
        self.display_titles.append("Original")
        self.Image = image

        self.compute_gradients()
        self.compute_covariance()
        self.compute_eigen()
        self.threshold_and_suppress(threshold)
        self.view(filename)

    def set_image(self,image):
        self.Image=image

    def get_gaussian_derivatives(self):
        return convolve(self.gaussian,_dx),convolve(self.gaussian,_dy)

    def compute_gradients(self):
        _dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        _dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        self.I_x = convolve(self.Image,_dx)
        self.I_y = convolve(self.Image,_dy)
        

    def compute_covariance(self):
        self.I_xx = convolve(self.I_x*self.I_x,self.gaussian)
        self.I_yy = convolve(self.I_y*self.I_y,self.gaussian)
        self.I_xy = convolve(self.I_x*self.I_y,self.gaussian)
        self.display_images.append(self.I_xx)
        self.display_titles.append("I_xx")
        self.display_images.append(self.I_yy)
        self.display_titles.append("I_yy")
        self.display_images.append(self.I_xy)
        self.display_titles.append("I_xy")
        
        self.M = np.zeros((self.image_height,self.image_width,2,2))
        
        for i in range(self.image_height):
            for j in range(self.image_width):
                self.M[i,j] = np.array([[self.I_xx[i, j], self.I_xy[i, j]], [self.I_xy[i, j],self.I_yy[i, j]]], dtype=np.float64)
        
    def compute_eigen(self):
        self.E = np.zeros((self.image_height,self.image_width))

        for i in range(self.image_height):
            for j in range(self.image_width):
                self.E[i,j] =  np.linalg.det(self.M[i,j]) - 0.04 * (np.power(np.trace(self.M[i,j]), 2))

    def threshold_and_suppress(self,T_ratio=0.1):
        print(self.E,T_ratio)
        T = T_ratio*self.E.max()
        self.thresholded = np.zeros((self.image_height,self.image_width)) + self.E
        for i in range(self.image_height):
            for j in range(self.image_width):
                if(self.thresholded[i][j]>T and self.check_8_neighbourhood(i,j)):
                    self.thresholded[i][j]=1
                else:
                    self.thresholded[i][j]=0

    def check_8_neighbourhood(self,row,col):
        step=[
              np.asarray([[-1,0],[1,0]]),
              np.asarray([[0,-1],[0,1]]),
              np.asarray([[-1,1],[1,-1]]),
              np.asarray([[-1,-1],[1,1]])
            ]
        index = np.asarray([row,col])
        valid_index=[]
        
        for i in range(4):
            for j in range(2):
                valid_index.append(index+step[i][j])
        
        for neighbourhood in valid_index:
            try:
                if self.E[neighbourhood[0]][neighbourhood[1]]>=self.E[index[0]][index[1]] :
                    return False
            except IndexError as e:
                        pass
        
        return True

    def view(self,filename):
        filename=filename.rpartition(".")[0]+'.png'
        pc, pr = np.where(self.thresholded == 1)
        plt.plot(pr, pc, 'r+')
        plt.imshow(self.Image, 'gray')
        plt.savefig('../results/harris/'+filename)
        plt.show()



