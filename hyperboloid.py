from manifold import Manifold
from metric import MinkowskiMetric
from numpy import arccosh, cosh, finfo, float64, isclose, logical_and, \
    ones_like, reshape, sinh, where, zeros_like

class Hyperboloid(Manifold):
    '''
        Hyperboloid manifolds. Assumes n-dimensional
        manifold is embedded in an (n+1)-dimensional ambient space.
    '''

    def __init__(self, n_dims):
        '''

        :param n_dims: dimensions of the manifold
        '''
        self.n_dims = n_dims
        self.metric = MinkowskiMetric(n_dims+1)

    def is_on_manifold(self, point):
        '''
        Determine whether point is in the set of manifold points.
        Should have point.point = -1 and point[0] > 0.
        :param point: (m, n_dims+1) np.array, representing m points on the
                        manifold
        :return: (m, 1) np.array of booleans
        '''
        dot_pp = self.metric.dot(point, point)
        return logical_and(
                            reshape(point[:, 0] > 0, (-1, 1)),
                            isclose(dot_pp, -ones_like(dot_pp))
        )

    def distance(self, u, v):
        '''
        Calculate the distance on the manifold between two points.
        Check if point on manifold, then insist on mini
        :param u, v:, (m, n_dims) np.arrays, each representing m vectors:
        :return: (m, 1) dimensional np.array, the distance between u and v
        '''
        # todo: implement check whether points on manifold or not
        neg_dot_uv = -self.metric.dot(u, v)
        return where(
                        neg_dot_uv > 1.,
                        arccosh(neg_dot_uv),
                        zeros_like(neg_dot_uv)
                    )

    def project_to_tangent_space(self, point, vector):
        '''
        Project vector into tangent space of point.
        Since point is on hyperboloid, point.point = -1
        :param point:  (m, n_dims) np.array, representing m points on the
                        hyperboloid:
        :param vector: (m, n_dims) np.array, representing m vectors in ]
                        (n+1)-dimensional ambient space
        :return: (m, n_dims) np.array, representing m vectors projected to
                tangent spaces of point
        '''
        return vector + self.metric.dot(point, vector)*point


    def exponential_map(self, point, v_TpS):
        '''
        Follow geodesic in direction v_TpS from point and
        return the resulting point.

        :param point: (m, n_dims+1) np.array,  m points on the hyperboloid:
        :param v_TpS: (m, n_dims+1) np.array, m vectors projected to tangent
                       spaces of point:
        :return: (m, n_dims+1) np.array, m points on the hyperboloid along the
                geodesic chosen by v_TpS
        '''
        # todo: check whether vector is in tangent space

        norm_v_TpS = self.metric.norm(v_TpS)
        # If v_TpS has zero norm, return the original point.
        # Correct behaviour and avoids division by zero in following calculation
        return where(
                        norm_v_TpS < finfo(float64).eps,
                        point,
                        cosh(norm_v_TpS) * point +
                                           sinh(norm_v_TpS) * (v_TpS/norm_v_TpS)
                     )

    def parallel_transport(self, vec_Tp0M, point_0, point_1):
        '''
        Parallel transport vector in tangent space of point 0 (Tp0M) to the
        tangent space of point 1 (Tp1M).
        :param vec_Tp0M: (m, n_dims+1) np.array, vector in Tp0M to transport
        :param point_0: initial point where vec_Tp0M
        :param point_1: final point to which vec_Tp0M will be transported
        :param n_steps: number of steps to break pole transport into
        :return: vec_Tp0M after parallel transport to point 1
        '''
        dirn = logarithmic_map(point_0, point_1)
        norm_dirn = self.metric.norm(dirn)
        unit_dirn = dirn / norm_dirn
        parallel_comp = self.metric.dot(vec_Tp0M, unit_dirn)
        vec_Tp1M = vec_Tp0M + parallel_comp * (
                 sinh(norm_dirn) * point_0 + (cosh(norm_dirn) - 1.) * unit_dirn)

        return vec_Tp1M