from preprocess import *
import matplotlib.pyplot as plt
from filter import *
import math
from scipy import ndimage
from scipy.ndimage.filters import convolve

class Canny():
    def __init__(self,image,filename='1'):

        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.thinned_image = np.zeros((self.image_height,self.image_width))
        self.thresholded_image = np.zeros((self.image_height,self.image_width))
        self.linked_image = np.zeros((self.image_height,self.image_width))
        self.display_images=[]
        self.display_titles=[]
        self.rgb = False
        self.gaussian = generate_gaussian_mask(5,1.4)

        self.Image = convolve(image,self.gaussian)

        self.strong_edge = 0.8
        self.weak_edge = 0.15
        self.angle_bin_size = 45
        self.compute_gradients()
        self.non_maximal_supression()
        self.threshold()
        self.hysterisis_link()
        self.view(filename)

    def set_image(self,image):
        self.Image=image

    def get_gaussian_derivatives(self):
        _dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        _dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        return convolve(self.gaussian,_dx),convolve(self.gaussian,_dy)

    def compute_gradients(self):
        self.G_dx, self.G_dy = self.get_gaussian_derivatives()
        image_dx = convolve(self.Image,self.G_dx)
        image_dy = convolve(self.Image,self.G_dy)
        self.grad_mag = np.sqrt(image_dx**2 + image_dy**2)
        self.grad_dir = np.arctan2(image_dy,image_dx)
        self.round_grad_dir()
    
    def threshold(self,T_low_ratio=0.5,T_high_ratio=1.33):
        T_high = T_high_ratio*np.mean(self.grad_mag)
        T_low  = T_low_ratio*T_high
        self.thresholded_image = self.thresholded_image + self.thinned_image
        self.thresholded_image[self.thresholded_image>=T_high]=self.strong_edge
        self.thresholded_image[self.thresholded_image< T_low]=0
        self.thresholded_image[(self.thresholded_image< T_high) & (self.thresholded_image>= T_low)]=self.weak_edge


    def in_range(self,index):
        i=index[0]
        j=index[1]
        return i<self.grad_mag.shape[0] and j<self.grad_mag.shape[1] and i>=0 and j>=0 

    def round_grad_dir(self):

        self.grad_dir = 180*self.grad_dir/(np.pi)
        self.grad_dir[self.grad_dir < 0] +=180 
        self.grad_dir[ (self.grad_dir<22.5)&(self.grad_dir>=0) ]        = 0
        self.grad_dir[ (self.grad_dir<67.5)&(self.grad_dir>=22.5) ]     = 45
        self.grad_dir[ (self.grad_dir<112.5)&(self.grad_dir>=67.5) ]    = 90    
        self.grad_dir[ (self.grad_dir<157.5)&(self.grad_dir>=112.5) ]   = 135
        self.grad_dir[ (self.grad_dir<=180)&(self.grad_dir>=157.5) ]    = 0

    def is_local_max(self,row,col,dir):
        
        index = np.asarray([row,col])
        dir = str(int(dir))
        step={
        '0'  :  np.asarray([[0,-1],[0,1]]),
        '90' :  np.asarray([[1,0],[-1,0]]),
        '45' :  np.asarray([[-1,1],[1,-1]]),
        '135':  np.asarray([[-1,-1],[1,1]])
        }

        check_index=[]
        for i in range(2):
            if(self.in_range(index+step[dir][i])):
                check_index.append(index+step[dir][i])
        
        local_maxima = index
        
        for idx in check_index:
            if(self.grad_mag[idx[0]][idx[1]]>self.grad_mag[local_maxima[0]][local_maxima[1]]):
                local_maxima = idx

        return np.array_equal(local_maxima,index)and(len(check_index)>0)

    def non_maximal_supression(self):
        output = np.array(self.grad_mag)
        for i in range(self.grad_mag.shape[0]):
            for j in range(self.grad_mag.shape[1]):
                if(self.is_local_max(i,j,self.grad_dir[i][j])):
                    output[i][j]=self.grad_mag[i][j]
                else:
                    output[i][j]=0
        
        self.thinned_image = self.thinned_image + output            


    def is_weak_pixel(self,pixel):
        return pixel==self.weak_edge

    def check_8_neighbourhood(self,row,col,img):
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
                if(self.in_range(index+step[i][j])):
                    valid_index.append(index+step[i][j])
        
        for neighbourhood in valid_index:
            if img[neighbourhood[0]][neighbourhood[1]]==self.strong_edge :
                return self.strong_edge
        
        return 0

    def hysterisis_link(self):
        
        self.linked_image = self.linked_image+self.thresholded_image

        for row in range(self.image_height):
            for col in range(self.image_width):
                if(self.is_weak_pixel(self.linked_image[row][col])):
                    self.linked_image[row][col] = self.check_8_neighbourhood(row,col,self.linked_image)
        


    def view(self,filename):
        filename=filename.rpartition(".")[0]+'.png'
        
        plt.figure(1)
        plt.imshow(self.grad_mag, 'gray')
        plt.savefig('../results/canny/grad_mag/'+filename)
        plt.show()

        plt.figure(1)
        plt.imshow(self.grad_dir, 'gray')
        plt.savefig('../results/canny/grad_dir/'+filename)
        plt.show()

        plt.figure(1)
        plt.imshow(self.thinned_image, 'gray')
        plt.savefig('../results/canny/thinned/'+filename)
        plt.show()

        plt.figure(1)
        plt.imshow(self.thresholded_image, 'gray')
        plt.savefig('../results/canny/thresholded/'+filename)
        plt.show()

        plt.figure(1)
        plt.imshow(self.linked_image, 'gray')
        plt.savefig('../results/canny/linked/'+filename)
        plt.show()

        plt.figure(1)
        plt.imshow(self.Image, 'gray')
        plt.savefig('../results/canny/original/'+filename)
        plt.show()

