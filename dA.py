import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import PIL.Image as Image

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


class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(
                np_rng.uniform(
                    low=- np.sqrt(2. / (n_hidden + n_visible)),
                    high= np.sqrt(2. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.x = input
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def rectify(self,X):
        return T.maximum(0,X)

    def get_hidden_values(self, input):
        return self.rectify(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.dot(hidden, self.W_prime) + self.b_prime

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        cost = T.mean((self.x-z)**2)

        #gparams = T.grad(cost, self.params)
        #updates = [
        #    (param, param - learning_rate * gparam)
        #    for param, gparam in zip(self.params, gparams)
        #]
        
        updates = stochasticGradient(cost, self.params,lr=learning_rate)

        return (cost, updates, y, self.get_reconstructed_input)

if __name__ == '__main__':
    pass
