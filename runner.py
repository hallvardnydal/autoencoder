import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from process import Process
from SdA              import SdA
    
def test_SdA(sample_size         = 60,
             finetune_lr         = 0.01, 
             pretraining_epochs  = 20,
             pretrain_lr         = 0.01, 
             training_epochs     = 100, 
             batch_size          = 30,
             corruption_levels   = [0.2],
             hidden_layers_sizes = [2000],
             img_size = (1020,1020),
             img_size_test = (600,1020)):
    
    process = Process()
    img_input,img_labels = process.read_in_images(["train-input"],["train-labels"])
    
    img_input = process.normalize(img_input)  
    #img_input = process.apply_clahe(img_input)
    #img_input = process.local_normalization(img_input)  
    
    img_input = img_input[:,:img_size[0],:img_size[1]]
    
    train_set  = img_input
    valid_set  = img_input[:1]
    test_set   = img_input[:1,:img_size_test[0],:img_size_test[1]]
    
    train_set_x,train_set_y = process.manipulate(train_set),train_set
    valid_set_x,valid_set_y = process.manipulate(valid_set),valid_set
    test_set_x,test_set_y   = process.manipulate(test_set),test_set
    
    train_set_x, table   =process.generate_set(train_set_x, sample_size = sample_size, stride = sample_size, img_size = img_size)
    valid_set_x, table   =process.generate_set(valid_set_x, sample_size = sample_size, stride = sample_size, img_size = img_size)
    test_set_x, table    =process.generate_set(test_set_x, sample_size = sample_size, stride = sample_size, img_size = img_size_test)
    train_set_y, table   =process.generate_set(train_set_y, sample_size = sample_size, stride = sample_size, img_size = img_size)
    valid_set_y, table   =process.generate_set(valid_set_y, sample_size = sample_size, stride = sample_size, img_size = img_size)
    test_set_y, table    =process.generate_set(test_set_y, sample_size = sample_size, stride = sample_size, img_size = img_size_test)
    
    train_set_x,train_set_y = train_set_x.astype(np.float32),train_set_y.astype(np.float32)
    valid_set_x,valid_set_y = valid_set_x.astype(np.float32),valid_set_y.astype(np.float32)
    test_set_x,test_set_y   = test_set_x.astype(np.float32), test_set_y.astype(np.float32)
    
    train_set_x, train_set_y = theano.shared(train_set_x,borrow=True),theano.shared(train_set_y,borrow=True)
    valid_set_x, valid_set_y = theano.shared(valid_set_x,borrow=True),theano.shared(valid_set_y,borrow=True)
    test_set_x, test_set_y   = theano.shared(test_set_x,borrow=True),theano.shared(test_set_y,borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    np_rng = np.random.RandomState()
    print '... building the model'

    sda = SdA(
        numpy_rng = np_rng,
        n_ins     = sample_size**2,
        n_outs    = sample_size**2,
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
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    
    datasets = [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model,output_fn = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    print '... finetuning of model'
    
    for n in xrange(training_epochs):
        costs = []
        for i in xrange(n_train_batches):
            costs.append(train_fn(i))
        
        cost = np.mean(costs)    
        #val_cost = validate_model()
        
        print "Epoch:",n,"Cost:",cost #,"Validation cost:",val_cost
    
    print "Test cost:", test_model()
     
    out = np.zeros((0,sample_size**2))
    for batch_index in xrange(train_set_x.get_value().shape[0]):
        out = np.vstack((out,output_fn(batch_index)))
        
    img_output = process.post_process(out, table, sample_size,img_shape=img_size_test)
        
    plt.figure()
    plt.imshow(test_set[0],cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(img_output[0],cmap=plt.cm.gray)
    
    xz = process.xz_stack(img_input)
    
    for m in xrange(xz.shape[0]):
        for n in xrange(xz.shape[1]):
            xz[m,n] = (xz[m,n]-xz[m,n].mean())/xz[m,n].std()
    
    xz_train, table    =process.generate_set(xz, sample_size = sample_size, stride = sample_size, img_size = img_size_test)
    xz_train = xz_train.astype(np.float32)
    test_set_x.set_value(xz_train)
    
    out = np.zeros((0,sample_size**2))
    for batch_index in xrange(train_set_x.get_value().shape[0]):
        out = np.vstack((out,output_fn(batch_index)))
        
    img_output = process.post_process(out, table, sample_size,img_shape=img_size_test)
    
    plt.figure()
    plt.imshow(xz[0],cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(img_output[0],cmap=plt.cm.gray)
    
    plt.show()
	
if __name__ == '__main__':
    test_SdA()
