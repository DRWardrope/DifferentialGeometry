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
                    [np.cos(0.5), np.sin(0.5)]
                ])
    expected = np.array([[0.5], [0.], [3.066370614359173]])

    circle = Sphere(1)
    assert_array_almost_equal(circle.distance(u, v), expected)

def test_project_to_tangent_space():
    p = np.array([
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.5*np.pi), np.sin(0.5*np.pi)],
                    [np.cos(np.pi), np.sin(np.pi)]
                ])
    v = np.array([
                    [0., 0.],
                    [1., 0.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.]
    ])
    expected = np.array([
                    [0., 0.],
                    [0., 0.],
                    [0., 1.],
                    [1., 0.],
                    [0., 1.]
    ])
    circle = Sphere(1)
    assert_array_almost_equal(circle.project_to_tangent_space(p, v), expected)

def test_exponential_map():
    p = np.array([
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.5*np.pi), np.sin(0.5*np.pi)],
                    [np.cos(np.pi), np.sin(np.pi)]
                ])
    v = np.array([
                        [0.,         0.],
                        [0.,         0.25*np.pi],
                        [0.25*np.pi, 0.],
                        [0.,         0.25*np.pi]
    ])
    expected = np.array([
                    [1., 0.],
                    [0.707106781186548, 0.707106781186548],
                    [0.707106781186548, 0.707106781186548],
                    [-0.707106781186548, 0.707106781186548]
    ])
    circle = Sphere(1)
    print("")
    print(circle.exponential_map(p, v))
    print(expected)
    assert_array_almost_equal(circle.exponential_map(p, v), expected)

def test_logarithmic_map():
    p0 = np.array([
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.5*np.pi), np.sin(0.5*np.pi)],
                    [np.cos(np.pi), np.sin(np.pi)]
                ])
    p1 = np.array([
                    [1., 0.],
                    [0.707106781186548, 0.707106781186548],
                    [0.707106781186548, 0.707106781186548],
                    [-0.707106781186548, 0.707106781186548]
    ])
    expected = np.array([
                        [0.,         0.],
                        [0.,         0.25*np.pi],
                        [0.25*np.pi, 0.],
                        [0.,         0.25*np.pi]
    ])

    circle = Sphere(1)
    print("")
    print(circle.logarithmic_map(p0, p1))
    print(expected)
    assert_array_almost_equal(circle.logarithmic_map(p0, p1), expected)

def test__pole_ladder_transport():
    p0 = np.array([
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.), np.sin(0.)],
                    [np.cos(0.5*np.pi), np.sin(0.5*np.pi)],
                    [np.cos(np.pi), np.sin(np.pi)]
                ])
    p1 = np.array([
                    [1., 0.],
                    [1., 0.],
                    [0.707106781186548, 0.707106781186548],
                    [np.cos(0.), np.sin(0.)]
    ])
    v = np.array([
                        [0.,         0.],
                        [0.,         0.25*np.pi],
                        [1, 0.],
                        [0.,         0.25*np.pi]
    ])
    expected = np.array([
                        [0.,         0.],
                        [0.,         0.25*np.pi],
                        [0.707106781186548, -0.707106781186548],
                        [0.,         0.75*np.pi]
    ])
    # Care should be taken when pole transporting between antipodes in one step!
    # v seems to lengthen!

    circle = Sphere(1)
    print("")
    print(circle._pole_ladder_transport(v, p0, p1))
    assert_array_almost_equal(circle._pole_ladder_transport(v, p0, p1), expected)

def test_parallel_transport():
    p0 = np.array([
        [np.cos(0.), np.sin(0.)],
        [np.cos(0.), np.sin(0.)],
        [np.cos(0.5 * np.pi), np.sin(0.5 * np.pi)],
        [np.cos(np.pi), np.sin(np.pi)]
    ])
    p1 = np.array([
        [1., 0.],
        [1., 0.],
        [0.707106781186548, 0.707106781186548],
        [np.cos(0.), np.sin(0.)]
    ])
    v = np.array([
        [0., 0.],
        [0., 0.25 * np.pi],
        [1, 0.],
        [0., 0.25 * np.pi]
    ])
    expected = np.array([
        [0., 0.],
        [0., 0.25 * np.pi],
        [0.707106781186548, -0.707106781186548],
        [0., 0.25 * np.pi]
    ])
    # Care should be taken when transporting to antipodes.
    # In this case, the sign of v_y is incorrect.

    circle = Sphere(1)
    print("")
    result = circle.parallel_transport(v, p0, p1)
    print(result)
    assert_array_almost_equal(result, expected)#, decimal=