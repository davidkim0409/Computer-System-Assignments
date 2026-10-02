/* CAS2107-01 Programming Assignment #1 - INT8 quantization.
 *
 * This is the only file you edit and the only file you submit. Implement the
 * six functions below in order. See include/quant.h for the contract and the
 * assignment PDF for the equations.
 *
 * Round with rintf(). You may NOT call roundf(), lroundf(), llroundf(),
 * round(), lround(), llround() or fesetround(): rintf() rounds halfway values
 * to the nearest even integer, which is what this assignment requires, while
 * roundf() rounds them away from zero.
 *
 * Clip before you cast. Converting an out-of-range float to an integer type is
 * undefined behavior in C, so the clamp must happen while the value is still a
 * float.
 *
 * Any extra function you write must be static.
 */

#include "quant.h"

#include <math.h>

/* ---- puzzle 1 ----------------------------------------------------------- */

QParams qp_from_minmax(float min, float max)
{
    (void)min; (void)max;

    /* TODO
     * Choose and return the quantization parameters for the range [min, max].
     */

    QParams p = {1.0f, 0};
    return p;
}

/* ---- puzzle 2 ----------------------------------------------------------- */

void quantize_tensor(const float *x, int8_t *q, size_t n, QParams p)
{
    (void)x; (void)q; (void)n; (void)p;

    /* TODO
     * Convert each Float32 element to an INT8 code using the supplied
     * quantization parameters.
     */
}

/* ---- puzzle 3 ----------------------------------------------------------- */

void dequantize_tensor(const int8_t *q, float *x, size_t n, QParams p)
{
    (void)q; (void)x; (void)n; (void)p;

    /* TODO
     * Convert each code back to Float32.
     */
}

/* ---- puzzle 4 ----------------------------------------------------------- */

void gemm_i8(const int8_t *A, const int8_t *B, int32_t *C,
             int M, int K, int N,
             int32_t a_zp, int32_t b_zp)
{
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
    (void)a_zp; (void)b_zp;

    /* TODO
     * Multiply two INT8 matrices. A is M×K, B is K×N, and C is M×N.
     * Everything is row-major: A[i][t] is A[i * K + t], B[t][j] is
     * B[t * N + j], and C[i][j] is C[i * N + j].
     */
}

/* ---- puzzle 5 ----------------------------------------------------------- */

void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N)
{
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;

    /* TODO
     * Multiply two FLOAT32 matrices. A is M×K, B is K×N, and C is M×N.
     * Everything is row-major (as in gemm_i8()).
     */
}

/* ---- puzzle 6 ----------------------------------------------------------- */

void requantize_tensor(const int32_t *acc, int8_t *q, size_t n,
                       float scale_a, float scale_b, QParams out)
{
    (void)acc; (void)q; (void)n; (void)scale_a; (void)scale_b; (void)out;

    /* TODO
     * Convert each INT32 accumulator to an INT8 output using the supplied
     * scales and output parameters.
     */
}
