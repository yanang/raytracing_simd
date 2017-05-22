#ifndef __RAY_MATH_TOOLKIT_H
#define __RAY_MATH_TOOLKIT_H

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

typedef struct _point3v{
    __m256d x;
    __m256d y;
    __m256d z;
}point3v;

__attribute__((always_inline)) static inline
void madd_vector(const point3v *a, const point3v *b, point3v *out)
{
    out->x = _mm256_add_pd(a->x, b->x);
    out->y = _mm256_add_pd(a->y, b->y);
    out->z = _mm256_add_pd(a->z, b->z);
}

__attribute__((always_inline)) static inline
void msubtract_vector(const point3v *a, const point3v *b, point3v *out)
{
    out->x = _mm256_sub_pd(a->x, b->x);
    out->y = _mm256_sub_pd(a->y, b->y);
    out->z = _mm256_sub_pd(a->z, b->z);
}

__attribute__((always_inline)) static inline
void mmultiply_vectors(const point3v *a, const point3v *b, point3v *out)
{
    out->x = _mm256_mul_pd(a->x, b->x);
    out->y = _mm256_mul_pd(a->y, b->y);
    out->z = _mm256_mul_pd(a->z, b->z);
}

__attribute__((always_inline)) static inline
void m_multiply_vector(const point3v *a, __m256d b, point3v *out)
{
    out->x = _mm256_mul_pd(a->x, b);
    out->y = _mm256_mul_pd(a->y, b);
    out->z = _mm256_mul_pd(a->z, b);
}

__attribute__((always_inline)) static inline
__m256d mdot_product(const point3v *a, const point3v *b)
{
    __m256d xmul = _mm256_mul_pd(a->x, b->x);
    __m256d ymul = _mm256_mul_pd(a->y, b->y);
    __m256d zmul = _mm256_mul_pd(a->z, b->z);
    xmul = _mm256_add_pd(xmul, ymul);
    xmul = _mm256_add_pd(zmul, xmul);
    return xmul;
}

__attribute__((always_inline)) static inline
void mcross_product(const point3v *a, const point3v *b, point3v *out)
{
    __m256d x1 = _mm256_mul_pd(a->y, b->z);
    __m256d x2 = _mm256_mul_pd(a->z, b->y);
    __m256d y1 = _mm256_mul_pd(a->z, b->x);
    __m256d y2 = _mm256_mul_pd(a->x, b->z);
    __m256d z1 = _mm256_mul_pd(a->x, b->y);
    __m256d z2 = _mm256_mul_pd(a->y, b->x);
    out->x = _mm256_sub_pd(x1, x2);
    out->y = _mm256_sub_pd(y1, y2);
    out->z = _mm256_sub_pd(z1, z2);

    /* clear version */
    /* out->x = __mm256_sub_pd(__mm256_mul_pd(a->y, b->z), __mm256_mul_pd(a->z, b->y)); */
    /* out->y = __mm256_sub_pd(__mm256_mul_pd(a->z, b->x), __mm256_mul_pd(a->x, b->z)); */
    /* out->z = __mm256_sub_pd(__mm256_mul_pd(a->x, b->y), __mm256_mul_pd(a->y, b->x)); */
}

__attribute__((always_inline)) static inline
void mnormalize(point3v *in)
{
    __m256d x2 = _mm256_mul_pd(in->x, in->x);
    __m256d y2 = _mm256_mul_pd(in->y, in->y);
    __m256d z2 = _mm256_mul_pd(in->z, in->z);
    __m256d t1 = _mm256_add_pd(x2, y2);
    t1 = _mm256_add_pd(z2, t1);
    t1 = _mm256_sqrt_pd(t1);
    in->x = _mm256_div_pd(in->x, t1);
    in->y = _mm256_div_pd(in->y, t1);
    in->z = _mm256_div_pd(in->z, t1);
}

__attribute__((always_inline)) static inline
__m256d mlength(const point3v *in)
{
    __m256d x2 = _mm256_mul_pd(in->x, in->x);
    __m256d y2 = _mm256_mul_pd(in->y, in->y);
    __m256d z2 = _mm256_mul_pd(in->z, in->z);
    __m256d t1 = _mm256_add_pd(x2, y2);
    t1 = _mm256_add_pd(z2, t1);
    t1 = _mm256_sqrt_pd(t1);
    return t1;
}

__attribute__((always_inline)) static inline
void mscalar_triple_product(const point3v *u, const point3v *v, const point3v *w,
                           point3v *out)
{
    mcross_product(v, w, out);
    mmultiply_vectors(u, out, out);
}

__attribute__((always_inline)) static inline
__m256d mscalar_triple(const point3v *u, const point3v *v, const point3v *w)
{
    point3v tmp;
    mcross_product(w, u, &tmp);
    return mdot_product(v, &tmp);
}

__attribute__((always_inline)) static inline
void normalize(double *v)
{
    double d = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    assert(d != 0.0 && "Error calculating normal");

    v[0] /= d;
    v[1] /= d;
    v[2] /= d;
}

__attribute__((always_inline)) static inline
double length(const double *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__attribute__((always_inline)) static inline
void add_vector(const double *a, const double *b, double *out)
{
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

__attribute__((always_inline)) static inline
void subtract_vector(const double *a, const double *b, double *out)
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

__attribute__((always_inline)) static inline
void multiply_vectors(const double *a, const double *b, double *out)
{
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
}

__attribute__((always_inline)) static inline
void multiply_vector(const double *a, double b, double *out)
{
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
}

__attribute__((always_inline)) static inline
void cross_product(const double *v1, const double *v2, double *out)
{
    out[0] = v1[1] * v2[2] - v1[2] * v2[1];
    out[1] = v1[2] * v2[0] - v1[0] * v2[2];
    out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

__attribute__((always_inline)) static inline
double dot_product(const double *v1, const double *v2)
{
    double dp=0;
    dp += v1[0] * v2[0];
    dp += v1[1] * v2[1];
    dp += v1[2] * v2[2];
    return dp;
}

__attribute__((always_inline)) static inline
void scalar_triple_product(const double *u, const double *v, const double *w,
                           double *out)
{
    cross_product(v, w, out);
    multiply_vectors(u, out, out);
}

__attribute__((always_inline)) static inline
double scalar_triple(const double *u, const double *v, const double *w)
{
    double tmp[3];
    cross_product(w, u, tmp);
    return dot_product(v, tmp);
}

#endif
