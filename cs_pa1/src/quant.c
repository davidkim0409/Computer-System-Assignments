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
    /* TODO
     * Choose and return the quantization parameters for the range [min, max].
     */
    
    // Widen the range to contain real zero
    float r_min = fmin(min, 0.0f);
    float r_max = fmax(max, 0.0f);

    // If widened range is single point, return s = 1, z = 0
    if (r_min == r_max) {
        QParams p = {1.0f, 0};
        return p;
    }

    // Compute s
    float s = (r_max - r_min) / 255.0f;

    // Even if range is not a single point,
    // if s is small enough, it can be rounded to zero.
    // In this case, return s = 1, z = 0
    if (s == 0.0f) {
        QParams p = {1.0f, 0};
        return p;
    }

    // Compute z
    float z = rintf(-128.0f - r_min / s);
    // Clamp z to say within [-128, 127]
    z = fmaxf(-128.0f, fminf(z, 127.0f));

    // Return p; type-cast z into int32_t
    QParams p = {s, (int32_t)z};
    return p;
}

/* ---- puzzle 2 ----------------------------------------------------------- */

void quantize_tensor(const float *x, int8_t *q, size_t n, QParams p)
{
    /* TODO
     * Convert each Float32 element to an INT8 code using the supplied
     * quantization parameters.
     */

    // For each element
    for (size_t i = 0; i < n; i++) {
        // Compute v = rintf(x_i/s) + z
        float v = rintf(x[i] / p.scale) + p.zero_point;

        // clip(v): saturate v into INT8 code range [-128, 127].
        v = fmaxf(-128.0f, fminf(v, 127.0f));

        // Convert into int8_t
        q[i] = (int8_t)v;
    }
}

/* ---- puzzle 3 ----------------------------------------------------------- */

void dequantize_tensor(const int8_t *q, float *x, size_t n, QParams p)
{
    /* TODO
     * Convert each code back to Float32.
     */

    // For each element
    for (size_t i = 0; i < n; i++) {
        // Compute c = q_i - z; use int32_t since difference can exceed INT8 range
        int32_t c = (int32_t)q[i] - p.zero_point;
        // x_i = c * s
        x[i] = c * p.scale;
    }
}

/* ---- puzzle 4 ----------------------------------------------------------- */

void gemm_i8(const int8_t *A, const int8_t *B, int32_t *C,
             int M, int K, int N,
             int32_t a_zp, int32_t b_zp)
{
    /*
     * Multiply two INT8 matrices. A is M×K, B is K×N, and C is M×N.
     * Everything is row-major: A[i][t] is A[i * K + t], B[t][j] is
     * B[t * N + j], and C[i][j] is C[i * N + j].
     */

    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;

            // sigma_t
            for (int t = 0; t < K; t++) {
                // Using A_it = A[i * K + t], B_tj = B[t * N + j],
                // a = A_it - z_a, b = B_tj - z_b
                int32_t a = (int32_t)A[(size_t)i * K + t] - a_zp;
                int32_t b = (int32_t)B[(size_t)t * N + j] - b_zp;

                // a * b and add to sum
                sum += a * b;
            }
            // C_ij = C[i * N + j] = sum
            C[(size_t)i * N + j] = sum;
        }
    }
}

/* ---- puzzle 5 ----------------------------------------------------------- */

void gemm_f32(const float *A, const float *B, float *C,
              int M, int K, int N)
{
    /*
     * Multiply two FLOAT32 matrices. A is M×K, B is K×N, and C is M×N.
     * Everything is row-major (as in gemm_i8()).
     */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            // sigma_t
            for (int t = 0; t < K; t++) {
                // A_it * B_tj = A[i * K + t] * B[t * N + j]
                sum += A[(size_t)i * K + t] * B[(size_t)t * N + j];
            }
            // C_ij = C[i * N +j] = sum
            C[(size_t)i * N + j] = sum;
        }
    }
}

/* ---- puzzle 6 ----------------------------------------------------------- */

void requantize_tensor(const int32_t *acc, int8_t *q, size_t n,
                       float scale_a, float scale_b, QParams out)
{
    /*
     * Convert each INT32 accumulator to an INT8 output using the supplied
     * scales and output parameters.
     */

    // Compute m = (s_a * s_b) / s_out
    float m = (scale_a * scale_b) / out.scale;

    // For each element
    for (size_t i = 0; i < n; i++) {
        // Compute v = rintf(m * acc_i) + z_out
        float v = rintf(m * (float)acc[i]) + out.zero_point;

        // clip
        v = fmaxf(-128.0f, fminf(v, 127.0f));

        // Convert to INT8
        q[i] = (int8_t)v;
    }
}
