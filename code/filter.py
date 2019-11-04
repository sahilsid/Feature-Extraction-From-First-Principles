
import numpy as np

def generate_gaussian_mask(size=5,sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
    
class Filter():
  def __init__(self,image,zero_padding='True'):
    self.zero_padding=False
    self.rgb = False
    self.filters = {
      'gaussian'  : np.asarray([[0.1019,0.1154,0.1019],[0.1154,0.1308,0.1154],[0.1019,0.1154,0.1019]],dtype=np.float32) ,
      'y_derivative'   : np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32),
      'x_derivative'   : np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32),
      'high_pass' : np.asarray([[0,-0.25,0],[-0.25,2,-0.25],[0,-0.25,0]]),
      'low_pass'  : np.asarray([[0,0.25,0],[0.25,-2,0.25],[0,0.25,0]])
    }
    self.kernel = np.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float32) 
    self.image=image
    self.channels = int(1)
    
    if(len(image.shape)==3 and image.shape[2]==3):
      self.rgb=True

    if(self.rgb):
        self.extract_luminance()    

    self.image = self.image[:,:,np.newaxis]

    self.image_height   = self.image.shape[0]
    self.image_width    = self.image.shape[1]
    self.zero_padding   = zero_padding

  
  def get_derivatives(self):
    self.set_kernel('x_derivative')
    x_Dr = self.my_2DConvolution()

    self.set_kernel('y_derivative')
    y_Dr = self.my_2DConvolution()

    return x_Dr,y_Dr

  def extract_luminance(self):
       image = np.asarray(self.image[:,:,0])
       image = 0.3*self.image[:,:,0] + 0.59*self.image[:,:,1]+ 0.11*self.image[:,:,2]
       self.image = image
    
  def set_kernel(self,filter='none'): 
    if(filter=='none'):
      self.kernel = np.asarray([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float32) 
    
    elif(not( (type(filter) is np.ndarray)) and filter in self.filters):
      self.kernel = self.filters[filter]
    
    elif( (type(filter) is np.ndarray) and (len(filter.shape)==2) or len(filter.shape)==3):
      self.kernel=filter

    else:
      print('\nInvalid Filter.\n')
      return False

    if(self.rgb and len(self.kernel.shape)==2):
      self.kernel = np.tile(self.kernel[:,:,None],[1,1,3])
    else:
      self.kernel=self.kernel[:,:,np.newaxis]
    return True

  
  def my_2DConvolution(self):
    filter=self.kernel
    
    f_height   = filter.shape[0]
    f_width    = filter.shape[1]

    filter = np.flip(filter)
    
    if(filter.shape[0]%2==0 or filter.shape[1]%2==0 ):
      print("Error : Filter size is even. \n")
      return  

    row_padding = int((f_height-1)/2)
    col_padding = int((f_width-1)/2)
    
    self.zero_padded_image = np.zeros((self.image_height+row_padding*2,self.image_width+col_padding*2,self.channels))
    self.zero_padded_image[row_padding:self.image_height+row_padding,col_padding:self.image_width+col_padding]=self.image
    
    output_image = np.zeros(self.image.shape)
    for row_pixel in range(row_padding,self.image_height+row_padding):
        for col_pixel in range (col_padding,self.image_width+col_padding):
            stepValue = np.sum(np.sum(np.multiply(filter,self.zero_padded_image[row_pixel-row_padding:row_pixel+row_padding+1,col_pixel-col_padding:col_pixel+col_padding+1]),axis = 0),axis = 0)
            output_image[row_pixel-row_padding][col_pixel-col_padding]= stepValue
              

    if(self.rgb==False):
      output_image=np.squeeze(output_image)

    return output_image

  def apply_filter(self,filter='none'):
    if(self.set_kernel(filter)):
      return self.my_2DConvolution()
