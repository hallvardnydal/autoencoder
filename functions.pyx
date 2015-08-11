import numpy as np
cimport numpy as np

cimport cython
@cython.boundscheck(False)

cdef inline average(np.ndarray[np.float64_t, ndim=2] array,unsigned int kernel_size, double product):
    cdef double average = 0.

    for m in xrange(kernel_size):
        for n in xrange(kernel_size):
            average += array[m,n]

    return average/product

cdef inline std_deviation(np.ndarray[np.float64_t, ndim=2] array,unsigned int kernel_size,double avg, double product):
    cdef double std = 0.

    for m in xrange(kernel_size):
        for n in xrange(kernel_size):
            std += (array[m,n]-avg)**2

    std = std/(product-1.)
    std = np.sqrt(std)
    return std


def local_normalization(np.ndarray[np.float64_t, ndim=3] img,int kernel_size):

    cdef unsigned int img_shape0,img_shape1,img_shape2
    img_shape0 = img.shape[0]
    img_shape1 = img.shape[1]
    img_shape2 = img.shape[2]

    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((img_shape0,img_shape1,img_shape2))

    assert img_shape1%kernel_size == 0
    assert img_shape2%kernel_size == 0

    cdef double product = kernel_size*kernel_size

    cdef np.ndarray[np.float64_t, ndim=2] ROI

    for m in xrange(img_shape0):
        for n in xrange(0,img_shape1,kernel_size):
            for k in xrange(0,img_shape2,kernel_size):
                ROI = img[m,n:(n+kernel_size),k:(k+kernel_size)]
                #avg = average(ROI,kernel_size,product)
                #std = std_deviation(ROI,kernel_size,avg,product)
                #img[m,n:(n+kernel_size),k:(k+kernel_size)] = (ROI-avg)/std
                out[m,n:(n+kernel_size),k:(k+kernel_size)] = (ROI-ROI.mean())/ROI.std()

    return out


