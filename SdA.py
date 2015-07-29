import time
import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from dA import dA

from collections import OrderedDict 

def stochasticGradient(cost,params,lr = 0.01):
    '''
    Stochastic Gradient Descent
    '''
    grads = T.grad(cost, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - lr * grad

    return updates

def apply_momentum(updates, params=None, momentum=0.9):
    """
    lasagne
    """

    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                    broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates


def momentum(cost, params, lr = 0.01, momentum=0.9):
    """
    lasagne
    """
    updates = stochasticGradient(cost, params, lr = lr)
    return apply_momentum(updates, momentum=momentum)


class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng = None,
        n_ins = 28*28,
        hidden_layers_sizes = None,
        n_outs = 10,
        corruption_levels = None):

        self.hidden_layers = []
        self.dA_layers     = []
        self.params        = []
        self.n_layers      = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        self.x = T.matrix('x')  
        self.y = T.matrix('y')  

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
        
        
        ###############
        # FINETUNING
        ###############
        
        self.last_layer = HiddenLayer(
            rng=numpy_rng,
            input=self.hidden_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs,
            last_layer = True
        )

        self.params.extend(self.last_layer.params)
        self.finetune_cost = self.last_layer.L2(self.y)
        self.output = self.last_layer.output

        self.errors = self.last_layer.L2(self.y)

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
        
    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        #gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        #updates = [
        #    (param, param - gparam * learning_rate)
        #    for param, gparam in zip(self.params, gparams)
        #]
        
        updates = stochasticGradient(self.finetune_cost, self.params,lr=learning_rate)

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )
        
        output_fn = theano.function(
            [index],
            self.output,
            givens={
                self.x: test_set_x[
                    index: (index + 1)
                ]
            },
            name='output'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return np.mean([valid_score_i(i) for i in xrange(n_valid_batches)])

        # Create a function that scans the entire test set
        def test_score():
            return np.mean([test_score_i(i) for i in xrange(n_test_batches)])

        return train_fn, valid_score, test_score,output_fn

if __name__ == '__main__':
    pass
