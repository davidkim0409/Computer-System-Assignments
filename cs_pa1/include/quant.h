#ifndef CAS2107_QUANT_H
#define CAS2107_QUANT_H

#include <stddef.h>
#include <stdint.h>

/* Quantization parameters for one tensor.
 *
 * A real value x is represented by an INT8 code q as
 *
 *     x ~= (q - z) * s,  where s = scale and z = zero_point
 *
 * zero_point is stored in an int32_t so that (int32_t)q - zero_point is safe,
 * but it is still a code and always stays inside [-128, 127]. */
typedef struct {
    float   scale;
    int32_t zero_point;
} QParams;

/* ---- puzzle 1 ----------------------------------------------------------- */

/* Choose quantization parameters that cover the range [min, max]. */
QParams qp_from_minmax(float min, float max);

/* ---- puzzle 2 ----------------------------------------------------------- */

/* Convert a Float32 tensor to INT8 using the supplied parameters. Values that
 * fall outside the represented range must be saturated. */
void quantize_tensor(const float *x, int8_t *q, size_t n, QParams p);

/* ---- puzzle 3 ----------------------------------------------------------- */

/* Convert an INT8 tensor back to Float32 using the supplied parameters. */
void dequantize_tensor(const int8_t *q, float *x, size_t n, QParams p);

/* ---- puzzle 4 ----------------------------------------------------------- */

/* Zero-point-corrected INT8 matrix multiplication. A is M x K, B is K x N,
 * and C is M x N. All matrices are row-major, and C uses INT32 accumulators. */
void gemm_i8(const int8_t *A, const int8_t *B, int32_t *C,
             int M, int K, int N,
             int32_t a_zp, int32_t b_zp);

/* ---- puzzle 5 ----------------------------------------------------------- */

/* Float32 matrix multiplication with the same dimensions and row-major layout
 * as gemm_i8. External BLAS or matrix libraries are not allowed. */
void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N);

/* ---- puzzle 6 ----------------------------------------------------------- */

/* Convert an INT32 accumulator tensor to INT8 using the input scales and the
 * supplied output quantization parameters. */
void requantize_tensor(const int32_t *acc, int8_t *q, size_t n,
                       float scale_a, float scale_b, QParams out);

#endif /* CAS2107_QUANT_H */
