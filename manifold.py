from numpy import finfo, float64, isclose, logical_and, maximum, where, zeros_like

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

    def distance(self, u, v):
        '''
        Calculate the distance on the manifold between two points.
        Check if point on manifold, then insist on mini
        :param u, v:, (m, n_dims) np.arrays, each representing m vectors:
        :return: (m, 1) dimensional np.array, the distance between u and v
        '''
        raise NotImplementedError("Should be implemented by subclass")

    def exponential_map(self, point, v_TpS):
        '''
        Follow geodesic in direction v_TpS from point and
        return the resulting point.

        :param point: (m, n_dims+1) np.array, representing m points on the
                        manifold:
        :param v_TpS: (m, n_dims+1) np.array, representing m vectors projected to
                tangent spaces of point:
        :return: (m, n_dims+1) np.array, m points on the manifold along the
                geodesic chosen by v_TpS, from point
        '''

        raise NotImplementedError("Should be implemented by subclass")

    def logarithmic_map(self, point0, point1):
        '''
        Inverse of exponential map

        :param point0: (m, n_dims) np.array, representing m "base" points:
        :param point1: (m, n_dims) np.array, representing m "target" points
        :return: (m, n_dims) np.array, m vectors in tangent spaces of point0,
                that would yield point1 if inserted in exponential map
        '''
        dot01 = self.metric.dot(point0, point1)
        v_Tp0M = point1 + dot01*point0
        dist = self.distance(point0, point1)
        norm_v_Tp0M = self.metric.norm(v_Tp0M)

        # If v_Tp0M has zero norm, return the original point.
        v_Tp0M_is_good = norm_v_Tp0M > finfo(float64).eps
            #self.is_in_tangent_space(point0, v_Tp0M)
        return where(
                        v_Tp0M_is_good,
                        v_Tp0M*(dist/norm_v_Tp0M),
                        v_Tp0M,
                     )
    def is_on_manifold(self, point):
        '''
        Determine whether point is in the set of manifold points
        :param point: (m, n_dims+1) np.array, representing m points on the
                        manifold
        :return: (m, 1) np.array of booleans
        '''
        raise NotImplementedError("Should be implemented by subclass")

    def is_in_tangent_space(self, point, vector):
        '''
        Determine whether vector is in the tangent space of manifold at point
        :param point: (m, n_dims+1) np.array, representing m points on the
                        manifold
        :param vector: (m, n_dims+1) np.array, representing m vectors
        :return: (m, 1) np.array of booleans
        '''
        dot_pv = self.metric.dot(point, vector)
        return isclose(dot_pv, zeros_like(dot_pv))

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
        dot_ratio = self.metric.dot(point, vector)/self.metric.dot(point, point)
        return vector - dot_ratio*point

    def _pole_ladder_transport(self, vec_Tp0M, point_0, point_1):
        '''
            Parallel transport of vector in tangent space of point 0 to the
            tangent space of point 1 using pole ladder algorithm defined in
            arxiv:1805.11436.
        :param vec_Tp0M: vector to be transported
        :param point_0: initial point where vec_Tp0M
        :param point_1: final point to which vec_Tp0M will be transported
        :return: vec_Tp0M after parallel transport to point 1
        '''
        # todo: check whether vector is in tangent space
        midpt_01 = self.exponential_map(
                                        point_0,
                                        0.5 * self.logarithmic_map(point_0, point_1)
                                        )
        prime_0 = self.exponential_map(point_0, vec_Tp0M)

        # Compute reflection of prime_0 on opposite side of midpoint
        prime_1 = self.exponential_map(
                                        midpt_01,
                                        -self.logarithmic_map(midpt_01, prime_0)
                                      )
        return -self.logarithmic_map(point_1, prime_1)

    def parallel_transport(self, vec_Tp0M, point_0, point_1, n_steps = 10):
        '''
        Parallel transport vector in tangent space of point 0 (Tp0M) to the tangent
        space of point 1 (Tp1M).
        :param vec_Tp0M: (m, n-dims+1) np.array, vector in Tp0M to be transported
        :param point_0: initial point where vec_Tp0M
        :param point_1: final point to which vec_Tp0M will be transported
        :param n_steps: number of steps to break pole transport into
        :return: vec_Tp0M after parallel transport to point 1
        '''
        # todo: check whether vector is in tangent space

        # Get rows that are actually changing in this batch. The others will stay the same.
        point_a = point_0.copy()
        vec_TpaM = vec_Tp0M.copy()

        for i in range(n_steps):
            point_b = self.exponential_map(
                    point_a,
                    (1 / (n_steps - i))*self.logarithmic_map(point_a, point_1)
                )
            vec_TpaM = self._pole_ladder_transport(vec_TpaM, point_a, point_b)
            point_a = point_b

        return vec_TpaM