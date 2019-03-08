from metric import EuclideanMetric, MinkowskiMetric
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_Metrics():
    g = EuclideanMetric(2)
    expected = np.array([
                            [1., 0],
                            [0., 1.]
                        ])
    assert_array_almost_equal(g.metric, expected)

    eta = MinkowskiMetric(2)
    expected = np.array([
                            [-1., 0],
                            [0., 1.]
                        ])
    assert_array_almost_equal(eta.metric, expected)

def test_dot():
    u = np.hstack([np.ones([2,1])]*3)
    v = np.hstack([np.zeros([2, 1]), np.ones([2,1]), -1.*np.ones([2,1])])

    g = EuclideanMetric(2)
    expected = np.array([[0.], [2.], [-2.]])
    assert_array_almost_equal(g.dot(u, v), expected)
    print(g.dot(u, v).shape)

    eta = MinkowskiMetric(2)
    expected = np.array([[0.], [0.],[0.]])
    assert_array_almost_equal(eta.dot(u, v), expected)

