from hyperboloid import Hyperboloid
import numpy as np
from numpy.testing import assert_array_almost_equal

def test_distance():
    u = np.array([
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(0.5), np.sinh(0.5)],
                    [np.cosh(10.), np.sinh(10.)]
                ])
    v = np.array([
                    [np.cosh(0.5), np.sinh(0.5)],
                    [np.cosh(0.5), np.sinh(0.5)],
                    [np.cosh(0.5), np.sinh(0.5)]
                ])
    expected = np.array([[0.5], [0.], [9.5]])

    hyperb = Hyperboloid(1)
    assert_array_almost_equal(hyperb.distance(u, v), expected)

# def test_project_to_tangent_space():
#     p = np.array([
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
#                     [np.cosh(np.pi), np.sinh(np.pi)]
#                 ])
#     v = np.array([
#                     [0., 0.],
#                     [1., 0.],
#                     [1., 1.],
#                     [1., 1.],
#                     [1., 1.]
#     ])
#     expected = np.array([
#                     [0., 0.],
#                     [0., 0.],
#                     [0., 1.],
#                     [1., 0.],
#                     [0., 1.]
#     ])
#     hyperb = Hyperboloid(1)
#     assert_array_almost_equal(hyperb.project_to_tangent_space(p, v), expected)

def test_is_on_manifold():
    p = np.array([
                    [np.cosh(0.), np.sinh(0.)],
                    [-1., 0],
                    [np.cosh(0.), np.sinh(0.)],
                    [3.14, 1.59],
                    [np.cosh(0.25*np.pi), np.sinh(0.25*np.pi)],
                    [np.cos(0.25 * np.pi), np.sin(0.25 * np.pi)],
                ])
    expected = np.array([
                    [True],
                    [False],
                    [True],
                    [False],
                    [True],
                    [False],
    ])
    hyperb = Hyperboloid(1)
    assert_array_almost_equal(hyperb.is_on_manifold(p), expected)

# def test_is_in_tangent_space():
#     p = np.array([
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.25*np.pi), np.sinh(0.25*np.pi)],
#                     [np.cosh(0.25 * np.pi), np.sinh(0.25 * np.pi)],
#                 ])
#     v = np.array([
#                     [0., 0.],
#                     [1., 0.],
#                     [0., 1.],
#                     [0., -1.],
#                     [1., 1.],
#                     [0.707106781186548,-0.707106781186548]
#     ])
#     expected = np.array([
#                     [True],
#                     [False],
#                     [True],
#                     [True],
#                     [False],
#                     [True],
#     ])
#     hyperb = Hyperboloid(1)
#     assert_array_almost_equal(hyperb.is_in_tangent_space(p, v), expected)
#
# def test_exponential_map():
#     p = np.array([
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
#                     [np.cosh(np.pi), np.sinh(np.pi)]
#                 ])
#     v = np.array([
#                         [0.,         0.],
#                         [0.,         0.25*np.pi],
#                         [0.25*np.pi, 0.],
#                         [0.,         0.25*np.pi]
#     ])
#     expected = np.array([
#                     [1., 0.],
#                     [0.707106781186548, 0.707106781186548],
#                     [0.707106781186548, 0.707106781186548],
#                     [-0.707106781186548, 0.707106781186548]
#     ])
#     hyperb = Hyperboloid(1)
#     print("")
#     print(hyperb.exponential_map(p, v))
#     print(expected)
#     assert_array_almost_equal(hyperb.exponential_map(p, v), expected)
#
# def test_logarithmic_map():
#     p0 = np.array([
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
#                     [np.cosh(np.pi), np.sinh(np.pi)]
#                 ])
#     p1 = np.array([
#                     [1., 0.],
#                     [0.707106781186548, 0.707106781186548],
#                     [0.707106781186548, 0.707106781186548],
#                     [-0.707106781186548, 0.707106781186548]
#     ])
#     expected = np.array([
#                         [0.,         0.],
#                         [0.,         0.25*np.pi],
#                         [0.25*np.pi, 0.],
#                         [0.,         0.25*np.pi]
#     ])
#
#     hyperb = Hyperboloid(1)
#     print("")
#     print(hyperb.logarithmic_map(p0, p1))
#     print(expected)
#     assert_array_almost_equal(hyperb.logarithmic_map(p0, p1), expected)
#
# def test__pole_ladder_transport():
#     p0 = np.array([
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.), np.sinh(0.)],
#                     [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
#                     [np.cosh(np.pi), np.sinh(np.pi)]
#                 ])
#     p1 = np.array([
#                     [1., 0.],
#                     [1., 0.],
#                     [0.707106781186548, 0.707106781186548],
#                     [np.cosh(0.), np.sinh(0.)]
#     ])
#     v = np.array([
#                         [0.,         0.],
#                         [0.,         0.25*np.pi],
#                         [1, 0.],
#                         [0.,         0.25*np.pi]
#     ])
#     expected = np.array([
#                         [0.,         0.],
#                         [0.,         0.25*np.pi],
#                         [0.707106781186548, -0.707106781186548],
#                         [0.,         0.75*np.pi]
#     ])
#     # Care should be taken when pole transporting between antipodes in one step!
#     # v seems to lengthen!
#
#     hyperb = Hyperboloid(1)
#     print("")
#     print(hyperb._pole_ladder_transport(v, p0, p1))
#     assert_array_almost_equal(hyperb._pole_ladder_transport(v, p0, p1), expected)
#
# def test_parallel_transport():
#     p0 = np.array([
#         [np.cosh(0.), np.sinh(0.)],
#         [np.cosh(0.), np.sinh(0.)],
#         [np.cosh(0.5 * np.pi), np.sinh(0.5 * np.pi)],
#         [np.cosh(np.pi), np.sinh(np.pi)]
#     ])
#     p1 = np.array([
#         [1., 0.],
#         [1., 0.],
#         [0.707106781186548, 0.707106781186548],
#         [np.cosh(0.), np.sinh(0.)]
#     ])
#     v = np.array([
#         [0., 0.],
#         [0., 0.25 * np.pi],
#         [1, 0.],
#         [0., 0.25 * np.pi]
#     ])
#     expected = np.array([
#         [0., 0.],
#         [0., 0.25 * np.pi],
#         [0.707106781186548, -0.707106781186548],
#         [0., 0.25 * np.pi]
#     ])
#     # Care should be taken when transporting to antipodes.
#     # In this case, the sign of v_y is incorrect.
#
#     hyperb = Hyperboloid(1)
#     print("")
#     result = hyperb.parallel_transport(v, p0, p1)
#     print(result)
#     assert_array_almost_equal(result, expected)#, decimal=