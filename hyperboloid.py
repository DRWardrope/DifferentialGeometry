
#
# def _parallel_transport(self, vec_Tp0M, point_0, point_1, n_steps=10):
#     dirn = logarithmic_map(point_0, point_1)
#     min_dot = float64(0. + margin) if 'float64' in K.floatx() else float32(
#         0. + margin)
#     norm_dirn = K.sqrt(tf.maximum(self.minkowski_tensor_dot(dirn, dirn), min_dot))
#     unit_dirn = dirn / norm_dirn
#     parallel_comp = self.minkowski_tensor_dot(vec_Tp0M, unit_dirn)
#     vec_TpaM = vec_Tp0M + parallel_comp * (
#             sinh(norm_dirn) * point_0 + (cosh(norm_dirn) - 1.) * unit_dirn)