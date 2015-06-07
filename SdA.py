import time
import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from dA import dA


class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng = None,
        n_ins = 28*28,
        hidden_layers_sizes = None,
        corruption_levels = None):

        self.hidden_layers = []
        self.dA_layers     = []
        self.params        = []
        self.n_layers      = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        self.x = T.matrix('x')  
        self.y = T.ivector('y')  

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.hidden_layers[-1].output

            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i])

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=hidden_layer.W,
                          bhid=hidden_layer.b)
            self.dA_layers.append(dA_layer)

    def pretraining_functions(self, train_set_x, batch_size):

        index = T.lscalar('index')  
        corruption_level = T.scalar('corruption') 
        learning_rate = T.scalar('lr')  
        
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        decoding_fns  = []
        for dA in self.dA_layers:
            cost, updates, y, decoding_fn = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            pretrain_fns.append(fn)
            decoding_fns.append(decoding_fn)
            
        output = y
        for n in xrange(len(decoding_fns)):
            n = -1 - n
            output = decoding_fns[n](output)
            
        output_fn = theano.function(
            [index,
             theano.Param(corruption_level, default=0.0)],
            output,
            givens={
                self.x: train_set_x[batch_begin: batch_end]
            }
        )

        return pretrain_fns, output_fn

if __name__ == '__main__':
    pass
