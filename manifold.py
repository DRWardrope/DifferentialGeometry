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

    def _pole_ladder_transport(self, vec_Tp0M, point_0, point_1,
                               exponential_map=None,
                               logarithmic_map=None):
        '''
            Parallel transport of vector in tangent space of point 0 to point 1
            using pole ladder algorithm defined in arxiv:1805.11436.
            Algorithm 2 is used, since this is reported to be more stable,
            although I'm not sure that applies here.
        :param vec_Tp0M: vector to be transported
        :param point_0: initial point where vec_Tp0M
        :param point_1: final point to which vec_Tp0M will be transported
        :return: vec_Tp1M: vec_Tp0M after parallel transport to point 1
        '''
        if exponential_map is None:
            exponential_map = self._exponential_map
        if logarithmic_map is None:
            logarithmic_map = self._logarithmic_map

        midpt_01 = exponential_map(point_0, 0.5 * logarithmic_map(point_0, point_1))
        prime_0 = exponential_map(point_0, vec_Tp0M)

        # Compute reflection of prime_0 on opposite side of midpoint
        prime_1 = exponential_map(midpt_01, -logarithmic_map(midpt_01, prime_0))
        return -logarithmic_map(point_1, prime_1)

    def _parallel_transport(self, vec_Tp0M, point_0, point_1, #indices=None,
                               exponential_map=None,
                               logarithmic_map=None,
                                use_pole_ladder=False,
                                n_steps = 10,
                                margin=1e-6
                            ):
        if exponential_map is None:
            exponential_map = self._exponential_map
        if logarithmic_map is None:
            logarithmic_map = self._logarithmic_map

        if use_pole_ladder:
            vec_TpaM = self._pole_ladder_transport(vec_Tp0M, point_0, point_1,
                                                   exponential_map=exponential_map, logarithmic_map=logarithmic_map)
        else:
            dirn = logarithmic_map(point_0, point_1)
            min_dot = float64(0. + margin) if 'float64' in K.floatx() else float32(0. + margin)
            norm_dirn = K.sqrt(tf.maximum(self.minkowski_tensor_dot(dirn, dirn), min_dot))
            unit_dirn = dirn/norm_dirn
            parallel_comp = self.minkowski_tensor_dot(vec_Tp0M, unit_dirn)
            vec_TpaM = vec_Tp0M + parallel_comp*(sinh(norm_dirn)*point_0 + (cosh(norm_dirn) - 1.)*unit_dirn)

        return vec_TpaM