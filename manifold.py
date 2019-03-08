class Manifold:
    '''
        Base class for (pseudo-)Riemannian manifolds. Assumes n-dimensional
        manifolds are embedded in an (n+1)-dimensional ambient space
    '''
    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the manifold
        '''
        self.n_dims = n_dims
        self.metric = None