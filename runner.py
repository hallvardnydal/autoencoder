import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from images_from_file import Process
from SdA              import SdA
    
def test_SdA(sample_size         = 64,
             finetune_lr         = 0.1, 
             pretraining_epochs  = 15,
             pretrain_lr         = 0.001, 
             training_epochs     = 1000,
             dataset             = 'mnist.pkl.gz', 
             batch_size          = 1,
             corruption_levels   = [0.2,0.2,0.2],
             hidden_layers_sizes = [1000, 750, 500]):
    
    process = Process()
    img_input,img_labels = process.read_in_images(["train-input"],["train-labels"])
    img_input = process.normalize(img_input)
    img_batched, table   =process.generate_set(img_input, sample_size = 64, stride = 64, img_size = (1024,1024))
    
    img_batched = img_batched.astype(np.float32)
    train_set_x = theano.shared(img_batched,borrow=True)
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    np_rng = np.random.RandomState(89677)
    print '... building the model'

    sda = SdA(
        numpy_rng = np_rng,
        n_ins     = sample_size**2,
        hidden_layers_sizes = hidden_layers_sizes
    )

    print '... Initializing pretraining functions'
    pretraining_fns, output_fn = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    
    print '... Layer-wise training of model'
    start_time = time.clock()

    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index      = batch_index,
                                            corruption = corruption_levels[i],
                                            lr         = pretrain_lr))
            print 'Layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('Layer-wise training ran for %.2fm' % ((end_time - start_time) / 60.))
    
    out = np.zeros((0,sample_size**2))
    for batch_index in xrange(n_train_batches):
        #print output_fn(batch_index)
        out = np.vstack((out,output_fn(batch_index)))
        
    img_output = process.post_process(out, table, sample_size)
        
    plt.figure()
    plt.imshow(img_input[0],cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(img_output[0],cmap=plt.cm.gray)
    plt.show()
	
if __name__ == '__main__':
    test_SdA()