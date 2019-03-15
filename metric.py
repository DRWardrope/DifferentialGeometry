from numpy import einsum, eye, reshape, sqrt

class Metric:
    '''
    Base class for metric
    '''

    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the space
        '''
        self.n_dims = n_dims
        self.metric = None

    def dot(self, u, v):
        '''
            Calculate dot_product for two vectors, u and v
            :param u, v: (m, n_dims) np.arrays, each representing m vectors
            :returns m, 1) u.v
        '''
        return reshape(einsum("ij,ai,aj->a", self.metric, u, v), (-1, 1))

    def norm(self, u):
        '''
        Calculate the norm of u
        :param u: (m, n_dims) np.array, representing m vectors:
        :return: (m, 1) dimensional np.array, representing the norm of u
        '''
        return sqrt(self.dot(u, u))


class EuclideanMetric(Metric):
    '''
        The Euclidean metric tensor (identity matrix)
    '''

    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the space
        '''
        self.n_dims = n_dims
        self.metric = eye(n_dims)


class MinkowskiMetric(Metric):
    '''
        The Minkowski metric tensor, satisfying convention that zeroth component
         is timelike: x.x = -(x^0)^2 + (x^i)^2
    '''

    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the space
        '''
        self.n_dims = n_dims
        self.metric = eye(n_dims)
        self.metric[0,0] = -1.