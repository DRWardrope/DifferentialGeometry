#import numpy as np
from manifold import Manifold
from metrics import EuclideanMetric
from np import arccos

class Sphere(Manifold):
    '''
        (Hyper-)Spherical manifolds. Assumes n-dimensional
        manifold is embedded in an (n+1)-dimensional ambient space
    '''
    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the manifold
        '''
        self.n_dims = n_dims
        self.metric = EuclideanMetric(n_dims)

    def distance(self, u, v):
        '''
        Calculate the distance on the manifold between two points
        :param u, v:, (m, n_dims) np.arrays, each representing m vectors:
        :return: (m, 1) dimensional np.array, the distance between u and v
        '''
        return arccos(self.metric.dot(u, v))

