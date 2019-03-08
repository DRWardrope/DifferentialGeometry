from numpy import diag, eye, reshape

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
        return reshape(diag(u.T.dot(self.metric.dot(v))), (-1, 1))


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