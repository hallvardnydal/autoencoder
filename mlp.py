import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    def __init__(self, 
            rng, 
            input, 
            n_in, 
            n_out, 
            params_dict = [None], 
            W = None, 
            b = None,
            activation = None,
            dropout_p = 0.0,
            maxoutsize = 1):
            
        if activation == None:
            activation = self.rectify

        self.srng    = RandomStreams(seed=234)
        self.input = input

        name = "W_hidden"
        if W is None :
            W_values = np.asarray(
                rng.uniform(
                    low  = -np.sqrt(2. / (n_in + n_out)),
                    high = np.sqrt(2. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name=name, borrow=True)

        n = 0
        for var in params_dict:
            try:
                if name == var.name:
                    W = params_dict[n]
                    print "Loading pre-trained parameter W"
            except:
                pass
            n += 1
     
        name = "b_hidden"
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name, borrow=True)

        n = 0
        for var in params_dict:
            try:
                if name == var.name:
                    b = params_dict[n]
                    print "Loading pre-trained parameter b"
            except:
                pass
            n += 1

        self.W = W
        self.b = b

        output = T.dot(input,self.W) + self.b

        # Maxout                                                                
        maxout_out = None                                                       
        for i in xrange(maxoutsize):                                            
            t = output[:,i::maxoutsize]                                   
            if maxout_out is None:                                              
                maxout_out = t                                                  
            else:                                                               
                maxout_out = T.maximum(maxout_out, t)  

        lin_output = self.dropout(maxout_out,dropout_p = dropout_p)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def rectify(self,X):
        return T.maximum(X,0)

    def dropout(self,X,dropout_p = 0.0):                                                  
        '''                                                                     
        Perform dropout with probability p                                      
        '''                                                                                                                                   
        retain_prob = 1 - dropout_p                                                   
        X *= self.srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
        X /= retain_prob                                                    
        return X 

    def TestVersion(self,rng,input,n_in,n_out,maxoutsize):
        return HiddenLayer(rng, input, n_in, n_out, W=self.W, b=self.b,
                 maxoutsize = maxoutsize, activation=None,dropout_p = 0.0)

if __name__ == '__main__':
    pass
