#import numpy as np
from manifold import Manifold
from metric import EuclideanMetric
from numpy import arccos, cos, finfo, float, sin, sqrt, where

class Sphere(Manifold):
    '''
        (Hyper-)Spherical manifolds. Assumes n-dimensional
        manifold is embedded in an (n+1)-dimensional ambient space.
        Assume radius of the sphere is 1.
    '''
    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the manifold
        '''
        self.n_dims = n_dims
        self.metric = EuclideanMetric(n_dims+1)

    def distance(self, u, v):
        '''
        Calculate the distance on the manifold between two points
        :param u, v:, (m, n_dims) np.arrays, each representing m vectors:
        :return: (m, 1) dimensional np.array, the distance between u and v
        '''
        return arccos(self.metric.dot(u, v))

    def norm(self, u):
        '''
        Calculate the norm of u
        :param u: (m, n_dims) np.array, representing m vectors:
        :return: (m, 1) dimensional np.array, representing the norm of u
        '''
        return sqrt(self.metric.dot(u, u))

    def project_to_tangent_space(self, point, vector):
        '''
        Project vector into tangent space of point.
        Since point is on hypersphere of radius 1, point.point = +1
        :param point:  (m, n_dims) np.array, representing m points on the
                        hypersphere:
        :param vector: (m, n_dims) np.array, representing m vectors in ]
                        (n+1)-dimensional ambient space
        :return: (m, n_dims) np.array, representing m vectors projected to
                tangent spaces of point
        '''
        return vector - self.metric.dot(point, vector)*point

    def exponential_map(self, point, v_TpS):
        '''
        Follow geodesic in direction v_TpS from point and
        return the resulting point.

        :param point: (m, n_dims) np.array, representing m points on the
                        hypersphere:
        :param v_TpS: (m, n_dims) np.array, representing m vectors projected to
                tangent spaces of point:
        :return: (m, n_dims) np.array, m points on the hypersphere along the
                geodesic chosen by v_TpS, from point
        '''
        norm_v_TpS = self.norm(v_TpS)
        # If v_TpS has zero norm, return the original point.
        # Correct behaviour and avoids division by zero in following calculation
        return where(
                        norm_v_TpS < finfo(float).eps,
                        point,
                        cos(norm_v_TpS) * point +
                                            sin(norm_v_TpS) * (v_TpS/norm_v_TpS)
                     )


    def logarithmic_map(self, point0, point1):
        '''
        Inverse of exponential map

        :param point0: (m, n_dims) np.array, representing m "base" points:
        :param point1: (m, n_dims) np.array, representing m "target" points
        :return: (m, n_dims) np.array, m vectors in tangent spaces of point0,
                that would yield point1 if inserted in exponential map
        '''
        dot01 = self.metric.dot(point0, point1)
        v_Tp0M = point1 - dot01*point0
        dist = self.distance(point0, point1)
        norm_v_Tp0M = self.norm(v_Tp0M)

        # If v_TpS has zero norm, return the original point.
        # Correct behaviour and avoids division by zero in following calculation
        return where(
                        norm_v_Tp0M < finfo(float).eps,
                        v_Tp0M,
                        v_Tp0M*dist/norm_v_Tp0M,
                     )
