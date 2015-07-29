import numpy as np
import matplotlib.pyplot as plt

from process import Process

def get_img():
    process = Process()
    img_input,img_labels = process.read_in_images(["train-input"],["train-labels"])
    img_input = process.normalize(img_input)
    return img_input
    
def get_xz_view(img):
    
    xz_view = np.zeros((0,1024))
    
    for n in xrange(img.shape[0]):
        xz_view = np.vstack((xz_view,img[n,0]))
        
    return xz_view

def plot_img(img):
    plt.figure()
    plt.imshow(img,cmap=plt.cm.gray)
     
def visualize():  
    img = get_img() 

    plot_img(img[0])
    
    xz_view = get_xz_view(img)
    
    plot_img(xz_view)
    plt.show()
    
    
	
if __name__ == "__main__":
	visualize()