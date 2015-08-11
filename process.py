import numpy as np
import glob
from PIL import Image
import cv2

class Process(object):

    def sort_key(self,input_file):                                                       
        return int(input_file.split('.')[0].split('_')[-1])  

    def img_to_array(self,files_input,files_labels,img_size):

        img_input = np.zeros((len(files_input),img_size[0]**2))
        img_labels = np.zeros((len(files_labels),img_size[0]**2))
        
        n_files = len(files_input)

        for n in xrange(n_files):
            File = files_input[n]
            img_temp = Image.open(File)                                                        
            img_temp = np.array(img_temp.getdata()).flatten(1)                                  
            img_input[n] = img_temp  

            File = files_labels[n]
            img_temp = Image.open(File)                                                        
            img_temp = np.array(img_temp.getdata()).reshape(img_temp.size)
            img_temp.flags.writeable = True

            img_labels[n] = img_temp.ravel()

        return img_input,img_labels
        
    def normalize(self,img):
        
        for n in xrange(img.shape[0]):
            img[n] -= img[n].mean()
            img[n] /= img[n].std()
        return img

    def init(self,directory_input,directory_labels):
        ''' 
        Function that reads in images from file and do
        some pre-processing
        '''

        files_input  = []
        files_labels = []

        image_groups = [0]
        counter = 0

        input_files = []
        for directory in directory_input:
            input_files += sorted(glob.glob(directory+"/*.tif"),key=self.sort_key)
        labeled_files = []
        for directory in directory_labels:
            labeled_files += sorted(glob.glob(directory+"/*.tif"),key=self.sort_key)

        return input_files, labeled_files 
        
    def read_in_images(self,directory_input,directory_labels,img_size = (1024,1024)):
        input_files, labeled_files = self.init(directory_input,directory_labels)
        img_input, img_labels = self.img_to_array(input_files,labeled_files,img_size)
        
        img_input  = img_input.reshape(img_input.shape[0],img_size[0],img_size[1])
        img_labels = img_labels.reshape(img_labels.shape[0],img_size[0],img_size[1])

        return img_input, img_labels
        
        
    def generate_set(self, images, sample_size, stride, img_size):
        
        assert img_size[0] % sample_size == 0
        number1 = img_size[0]/sample_size
        number2 = img_size[1]/sample_size
        
        img_batched = np.zeros((images.shape[0]*number1*number2, sample_size**2))
        table       = np.zeros((images.shape[0]*number1*number2, 3))

        table_number = 0
        for img_number in xrange(images.shape[0]):
            for n in xrange(number1):
                for m in xrange(number2):

                    img_start_y = stride*n
                    img_end_y   = stride*n + sample_size
                    img_start_x = stride*m
                    img_end_x   = stride*m + sample_size

                    img_batched[table_number,:] = images[img_number,img_start_y:img_end_y,img_start_x:img_end_x].reshape(sample_size**2)  

                    table[table_number,0] = img_number
                    table[table_number,1] = img_start_y
                    table[table_number,2] = img_start_x
                    table_number += 1

        return img_batched,table
        
    def post_process(self,img_batched, table, sample_size, img_shape = (1024,1024)):
    
        nr_images    = np.max(table[:,0]) + 1 
        img_batched  = img_batched.reshape(img_batched.shape[0],sample_size,sample_size)  
               
        img      = np.zeros((nr_images, img_shape[0], img_shape[1]))
        count    = np.zeros((nr_images, img_shape[0], img_shape[1]))
        
        for i in xrange(table.shape[0]):       
            img[table[i,0],(table[i,1]):(table[i,1]+sample_size),(table[i,2]):(table[i,2] + sample_size)]      += img_batched[i]      
            count[table[i,0],(table[i,1]):(table[i,1]+sample_size),(table[i,2]):(table[i,2] + sample_size)] += np.ones((sample_size,sample_size))                                              

        count = count.astype(np.float32)

        img      /= count

        return img
        
    def average(self,img):
        img_avg = np.zeros(img.shape)
        avg = np.mean(img,axis=0)
    
        for n in xrange(img.shape[0]):
            img_avg[n] = avg 

        return img_avg
        
    def local_normalization(self,img,kernel = (500,500)):
        img_avg = np.zeros(img.shape)
    
        for n in xrange(img.shape[0]):
            for m in xrange(img.shape[1]-kernel[0]):
                for k in xrange(img.shape[2]-kernel[1]):
                    ROI = img_avg[n,m:m+kernel[0],k:k+kernel[1]]
                    img_avg[n,m:m+kernel[0],k:k+kernel[1]] = (ROI-ROI.mean())/ROI.std()

        return img_avg
        
    def gaussian_average(self,img,stride,sigma=1):
        weight = np.arange(stride)
        weight -= weight.mean()
        weight = (1/(2.*np.pi*sigma))*np.exp((-weight**2)/(2*sigma**2))
        weight /= weight.sum()
        
        img_avg = np.zeros(img.shape)      
        avg = np.zeros(img.shape[1])
        
        for n in xrange(img.shape[0]):
            avg += img[n]*weight[n]

        for n in xrange(img.shape[0]):
            img_avg[n] = avg 

        return img_avg
        
    def apply_clahe(self,img_input):
        img_input = self.normalize(img_input)
        img_input -= img_input.min()
        img_input /= img_input.max()
        img_input *= 255.
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(15,15))

        clahe_normalized = np.zeros(img_input.shape)
        for n in xrange(img_input.shape[0]):
            clahe_normalized[n] =  clahe.apply(img_input[n].astype(np.uint8))
        
        return self.normalize(clahe_normalized)
    
    def pick_one(self,img,stride):
        one = np.zeros(img.shape)
        
        for n in xrange(img.shape[0]):
            one[n] = img[(stride/2)-1]
        return img
        
    def manipulate(self,img,stride=6):
        img_avg = np.zeros(img.shape)
        
        for m in xrange(img.shape[0]):
            for n in xrange(0,img.shape[1],stride):
                img_avg[m,n:n+stride] = self.average(img[m,n:n+stride])
        #for m in xrange(img.shape[0]):
        #    for n in xrange(0,img.shape[1],stride):
        #        img_avg[m,n:n+stride] = self.gaussian_average(img[m,n:n+stride],stride)
        #for m in xrange(img.shape[0]):
        #    for n in xrange(0,img.shape[1],stride):
        #        img_avg[m,n:n+stride] = self.pick_one(img[m,n:n+stride],stride)
        
        return img_avg
        
    def expand(self,i,stride):
        out = np.zeros((stride,i.shape[0]))
        for n in xrange(stride):
            out[n] = i
        return out
        
    def xz_stack(self,stack,images=2,stride=6):
        xz_stack = stack[:,:images,:]

        xz_avg = np.zeros((xz_stack.shape[1],xz_stack.shape[0]*stride,xz_stack.shape[2]))
        
        for n in xrange(xz_avg.shape[0]):
            for m in xrange(xz_stack.shape[0]):
                xz_avg[n,stride*m:stride*(m+1)] = self.expand(xz_stack[m,n],stride)
                
        return xz_avg
            