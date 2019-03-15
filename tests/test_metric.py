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
    u = np.vstack([np.ones([1,2])]*3)
    v = np.vstack([np.zeros([1, 2]), np.ones([1,2]), -1.*np.ones([1,2])])

    g = EuclideanMetric(2)
    expected = np.array([[0.], [2.], [-2.]])
    print("")
    print(u)
    print(v)
    print(g.dot(u, v))
    assert_array_almost_equal(g.dot(u, v), expected)

    eta = MinkowskiMetric(2)
    expected = np.array([[0.], [0.],[0.]])
    assert_array_almost_equal(eta.dot(u, v), expected)

def test_norm():
    v = np.vstack([np.zeros([1, 2]), np.ones([1,2]), -1.*np.ones([1,2])])

    g = EuclideanMetric(2)
    expected = np.array([[0.], [np.sqrt(2.)], [np.sqrt(2.)]])
    print("")
    print(v)
    assert_array_almost_equal(g.norm(v), expected)

    eta = MinkowskiMetric(2)
    expected = np.array([[0.], [0.],[0.]])
    assert_array_almost_equal(eta.norm(v), expected)

