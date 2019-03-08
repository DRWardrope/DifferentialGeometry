from sphere import Sphere
import numpy as np
from numpy.testing import assert_array_almost_equal

def test_distance():
    u = np.array([
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.5), np.sin(0.5)],
                    [np.cos(10.), np.sin(10.)]
                ])
    v = np.array([
                    [np.cos(0.5), np.sin(0.5)],
                    [np.cos(0.5), np.sin(0.5)],
                    [np.cos(0.5.), np.sin(0.5)]
                ])
    expected = np.array([[0.5], [0.], [3.216814692820414]])

    circle = Sphere(1)
    assert_array_almost_equal(circle.distance(u, v), expected)