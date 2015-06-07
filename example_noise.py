import numpy as np
import matplotlib.pyplot as plt
from images_from_file import Process

def example_noise():
    process = Process()
    img_input,img_labels = process.read_in_images(["train-input"],["train-labels"])
    img_input = process.normalize(img_input)
    
    img = img_input[0]
    
    img_out = img *np.random.binomial(1,0.8,size = (1024,1024))
    
    plt.figure()
    plt.imshow(img,cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(img_out,cmap=plt.cm.gray)
    plt.show()
    
if __name__ == '__main__':
    example_noise()