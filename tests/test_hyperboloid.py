from hyperboloid import Hyperboloid
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


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


def test_is_in_tangent_space():
    p = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.5 * np.pi), np.sinh(0.5 * np.pi)],
        [np.cosh(np.pi), np.sinh(np.pi)],
    ])
    v = np.array([
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [0., -1.],
        [0.478393040868114, 0.521606959131886],
        [0.499066278634146, 0.500933721365854]
    ])
    expected = np.array([
        [True],
        [False],
        [True],
        [True],
        [True],
        [True],
    ])
    hyperb = Hyperboloid(1)
    assert_array_almost_equal(hyperb.is_in_tangent_space(p, v), expected)

def test_project_to_tangent_space():
    p = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
        [np.cosh(np.pi), np.sinh(np.pi)]
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
        [0.478393040868114, 0.521606959131886],
        [0.499066278634146, 0.500933721365854]
    ])
    hyperb = Hyperboloid(1)
    v_TpM = hyperb.project_to_tangent_space(p, v)
    assert_array_almost_equal(v_TpM, expected)
    # projection(projection(v)) should equal projection(v)
    assert_array_almost_equal(
                                hyperb.project_to_tangent_space(p, v_TpM),
                                expected
    )

def test_exponential_map():
    p = np.array([
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(0.5*np.pi), np.sinh(0.5*np.pi)],
              #      [np.cosh(np.pi), np.sinh(np.pi)]
                ])
    v = np.array([
                    [0.,         0.],
                    [0.,         0.25*np.pi],
                    [0.478393040868114, 0.521606959131886],
             #       [0.499066278634146, 0.500933721365854]
    ])
    expected = np.array([
                    [1., 0.],
                    [1.324609089252006, 0.86867096148601],
                    [3.04543575, 2.87657416],
    ])
    hyperb = Hyperboloid(1)
    print("")
    print(hyperb.exponential_map(p, v))
    print(expected)
    assert_array_almost_equal(hyperb.exponential_map(p, v), expected)

def test_logarithmic_map():
    # Leverage trusted exponential map to verify logarithmic map
    p0 = np.array([
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(1), np.sinh(1)],
                    [np.cosh(1), np.sinh(1)],
                ])
    p1 = np.array([
                    [np.cosh(0.), np.sinh(0.)],
                    [np.cosh(1.), np.sinh(1.)],
                    [np.cosh(-1.), np.sinh(-1.)],
                    [np.cosh(-np.pi), np.sinh(-np.pi)]
    ])

    hyperb = Hyperboloid(1)
    vectors = hyperb.logarithmic_map(p0, p1)
    p1_from_v = hyperb.exponential_map(p0, vectors)
    assert_array_equal(
                        hyperb.is_on_manifold(p1_from_v),
                        np.ones((p1_from_v.shape[0], 1), dtype=bool)
    )
    assert_array_almost_equal(p1, p1_from_v)

def test__pole_ladder_transport():
    # Test relies on fact that parallel transporting vector from p0->p1->p0 along
    # same path should yield the same vector in Tp0M
    p0 = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.5 * np.pi), np.sinh(0.5 * np.pi)],
        [np.cosh(np.pi), np.sinh(np.pi)]
    ])
    p1 = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(1.), np.sinh(1.)],
        [np.cosh(-1.), np.sinh(-1.)],
        [np.cosh(-np.pi), np.sinh(-np.pi)]
    ])
    v_Tp0M = np.array([
        [0., 0.],
        [0., 1.],
        [0.478393040868114, 0.521606959131886],
        [0.499066278634146, 0.500933721365854]
    ])

    hyperb = Hyperboloid(1)
    print("")
    v_Tp1M = hyperb._pole_ladder_transport(v_Tp0M, p0, p1)
    print(v_Tp1M)
    v_Tp0M_ptd = hyperb._pole_ladder_transport(v_Tp1M, p1, p0)
    print(v_Tp0M_ptd)
    assert_array_almost_equal(v_Tp0M, v_Tp0M_ptd)

def test_parallel_transport():
    # Test relies on fact that parallel transporting vector from p0->p1->p0 along
    # same path should yield the same vector in Tp0M
    p0 = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(0.5 * np.pi), np.sinh(0.5 * np.pi)],
        [np.cosh(np.pi), np.sinh(np.pi)]
    ])
    p1 = np.array([
        [np.cosh(0.), np.sinh(0.)],
        [np.cosh(1.), np.sinh(1.)],
        [np.cosh(-1.), np.sinh(-1.)],
        [np.cosh(-np.pi), np.sinh(-np.pi)]
    ])
    v_Tp0M = np.array([
        [0., 0.],
        [0., 1.],
        [0.478393040868114, 0.521606959131886],
        [0.499066278634146, 0.500933721365854]
    ])

    hyperb = Hyperboloid(1)
    print("")
    v_Tp1M = hyperb.parallel_transport(v_Tp0M, p0, p1)
    print(v_Tp1M)
    v_Tp0M_ptd = hyperb.parallel_transport(v_Tp1M, p1, p0)
    print(v_Tp0M_ptd)
    assert_array_almost_equal(v_Tp0M, v_Tp0M_ptd)

