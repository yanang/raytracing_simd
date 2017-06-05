#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "math-toolkit.h"
#include "primitives.h"
#include "raytracing.h"
#include "idx_stack.h"
#include <immintrin.h>

#define MAX_REFLECTION_BOUNCES	3
#define MAX_DISTANCE 1000000000000.0
#define MIN_DISTANCE 0.00001
#define SAMPLES 4

#define SQUARE(x) (x * x)
#define MAX(a, b) (a > b ? a : b)

#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_NGT_US   0x0a /* Not-greater-than (unordered, signaling)  */

typedef struct _rgb {
    __m256d r;
    __m256d g;
    __m256d b;
} rgb;


static inline void COPY_POINT3v(point3v *out,const point3 in)
{
    out->x=_mm256_set1_pd(in[0]);
    out->y=_mm256_set1_pd(in[1]);
    out->z=_mm256_set1_pd(in[2]);
}

static inline void COPY_RGB(point3v *out,const point3 in)
{
    out->x=_mm256_set1_pd(in[0]);
    out->y=_mm256_set1_pd(in[1]);
    out->z=_mm256_set1_pd(in[2]);
}

void COPY_POINT3vv(point3v a, const point3v b)
{
    __m256d mzero = _mm256_setzero_pd();
    a->x = _mm256_add_pd(mzero, b->x);
    a->y = _mm256_add_pd(mzero, b->y);
    a->z = _mm256_add_pd(mzero, b->z);
}
/* @param t t distance
 * @return 1 means hit, otherwise 0
 */

static __m256d raySphereIntersection(const point3v ray_e,
                                     const point3v ray_d,
                                     const sphere *sph,
                                     intersection *ip, __m256d *t1)
{
    point3v l;
    point3v sphcen;
    COPY_POINT3v(sphcen, sph->center);
    subtract_vector(&sphcen, ray_e, &l);
    __m256d ms = dot_product(l, ray_d);
    __m256d ml2 = dot_product(l, l);

    point3v sphrad;
    COPY_POINT3v(sphrad, sph->radius);
    __m256d mr2 = _mm256_mul_pd(sphrad, sphrad);

    __m256d mzero = _mm256_setzero_pd();
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */
    __m256d if1 = _mm256_cmp_pd(ms, mzero, _CMP_LE_OS);
    __m256d if2 = _mm256_cmp_pd(ml2, mr2, _CMP_GT_OS);
    if1 = _mm256_and_pd(if1, if2); // if (s < 0 && l2 > r2)

    __m256d mm2 = _mm256_mul_pd(ms, ms);
    mm2 = _mm256_sub_pd(l2, ms); // mm2 = ml2 - ms * ms

    if2 = _mm256_cmp_pd(mm2, mr2, _CMP_GT_OS); // if (m2 > r2)

    __m256d mq = _mm256_sub_pd(mr2, mm2);
    mq = _mm256_sqrt_pd(mq);

    __m256d if3 = _mm256_cmp_pd(ml2, mr2, _CMP_GT_OS); // if (l2 > r2)
    __m256d smiq = _mm256_sub_pd(ms, mq); // s-q value
    __m256d sadq = _mm256_add_pd(ms, mq); // s+q value

    __m256i iall1 = _mm256_set1_epi64x(-1); // iall1 = 0xFF...F
    n   __m256d dall1 = _mm256_castsi256_pd(iall1);

    if1 = _mm256_or_pd(if1, if2); // if1 = if1 or if2, after will use it to load the t1 before
    __m256d notif1 = _mm256_xor_pd(if1, dall1); // notif1 = not ( if1 )

    __m256d t1copy = _mm256_add_pd(*t1, mzero); // create a copy for *t1

    smiq = _mm256_and_pd(smiq, if3);
    __m256d notif3 = _mm256_xor_pd(if3, dall1); // notif3 = not ( if3 )
    sadq = _mm256_and_pd(sadq, notif3);
    __m256d newt1 = _mm256_or_pd(smiq, sadq); // the new value of *t1 if not return

    t1copy = _mm256_and_pd(t1copy, if1);
    t1notre = _mm256_and_pd(newt1, notif1);
    __m256d t1new = _mm256_add_pd(t1copy, t1notre);
    *t1 = __m256d_add_pd(mzero, t1new);

    multiply_vector(ray_d, *t1, ip->point);
    add_vector(ray_e, ip->point, ip->point);

    subtract_vector(ip->point, sphcen, ip->normal);
    normalize(ip->normal);

    __m256d dotres = dot_product(ip->normal, ray_d);
    __m256d ifdot = _mm256_cmp_pd(dotres, mzero, _CMP_GT_OS); // if dotres greater than 0
    __m256d notifdot = _mm256_xor_pd(ifdot, dall1);
    __m256d ipcopy = _mm256_add_pd(ip->normal, mzero);
    ipcopy = _mm256_and_pd(ipcopy, notifdot);
    __m256d minus1 = _mm256_set1_pd(-1);
    multiply_vector(ip->normal, minus1, ip->normal);
    ip->normal = _mm256_and_pd(ip->normal, ifdot);
    ip->normal = _mm256_or_pd(ip->normal, ipcopy);
}

/* @return 1 means hit, otherwise 0; */
static int rayRectangularIntersection(const point3 ray_e,
                                      const point3 ray_d,
                                      rectangular *rec,
                                      intersection *ip, double *t1)
{
    point3 e01, e03, p;
    subtract_vector(rec->vertices[1], rec->vertices[0], e01);
    subtract_vector(rec->vertices[3], rec->vertices[0], e03);

    cross_product(ray_d, e03, p);

    double det = dot_product(e01, p);

    /* Reject rays orthagonal to the normal vector.
     * I.e. rays parallell to the plane.
     */
    if (det < 1e-4)
        return 0;

    double inv_det = 1.0 / det;

    point3 s;
    subtract_vector(ray_e, rec->vertices[0], s);

    double alpha = inv_det * dot_product(s, p);

    if ((alpha > 1.0) || (alpha < 0.0))
        return 0;

    point3 q;
    cross_product(s, e01, q);

    double beta = inv_det * dot_product(ray_d, q);
    if ((beta > 1.0) || (beta < 0.0))
        return 0;

    *t1 = inv_det * dot_product(e03, q);

    if (alpha + beta > 1.0f) {
        /* for the second triangle */
        point3 e23, e21;
        subtract_vector(rec->vertices[3], rec->vertices[2], e23);
        subtract_vector(rec->vertices[1], rec->vertices[2], e21);

        cross_product(ray_d, e21, p);

        det = dot_product(e23, p);

        if (det < 1e-4)
            return 0;

        inv_det = 1.0 / det;
        subtract_vector(ray_e, rec->vertices[2], s);

        alpha = inv_det * dot_product(s, p);
        if (alpha < 0.0)
            return 0;

        cross_product(s, e23, q);
        beta = inv_det * dot_product(ray_d, q);

        if ((beta < 0.0) || (beta + alpha > 1.0))
            return 0;

        *t1 = inv_det * dot_product(e21, q);
    }

    if (*t1 < 1e-4)
        return 0;

    COPY_POINT3(ip->normal, rec->normal);
    if (dot_product(ip->normal, ray_d)>0.0)
        multiply_vector(ip->normal, -1, ip->normal);
    multiply_vector(ray_d, *t1, ip->point);
    add_vector(ray_e, ip->point, ip->point);

    return 1;
}

static void localColor(color local_color,
                       const color light_color, double diffuse,
                       double specular, const object_fill *fill)
{
    color ambi = { 0.1, 0.1, 0.1 };
    color diff, spec, lightCo, surface;

    /* Local Color = ambient * surface +
     *               light * ( kd * surface * diffuse + ks * specular)
     */

    COPY_COLOR(diff, fill->fill_color);
    multiply_vector(diff, fill->Kd, diff);
    multiply_vector(diff, diffuse, diff);
    COPY_COLOR(lightCo, light_color);
    multiply_vectors(diff, lightCo, diff);

    COPY_COLOR(spec, light_color);
    multiply_vector(spec, fill->Ks, spec);
    multiply_vector(spec, specular, spec);

    COPY_COLOR(surface, fill->fill_color);
    multiply_vectors(ambi,surface, ambi);
    add_vector(diff, ambi, diff);
    add_vector(diff, spec, diff);
    add_vector(local_color, diff, local_color);
}

/* @param d direction of the ray into intersection
 * @param l direction of intersection to light
 * @param n surface normal
 */
static void compute_specular_diffuse(__m256d *diffuse,
                                     __m256d *specular,
                                     const point3v d, const point3v l,
                                     const point3v n, double phong_pow)
{
    point3v d_copy, l_copy, middle, r;

    __m256d minus1 = _mm256_set1_pd(-1);
    __m256d two = _mm256_set1_pd(2);

    COPY_POINT3vv(d_copy, d);
    multiply_vector(d_copy, minus1, d_copy);
    normalize(d_copy);

    COPY_POINT3vv(l_copy, l);
    multiply_vector(l_copy, minus1, l_copy);
    normalize(l_copy);

    __m256d tmp = dotproduct(n, l_copy);
    multiply_vector(n, tmp, middle);
    multiply_vector(middle, two, middle);
    subtract_vector(middle, l_copy, r);
    normalize(r);

    __m256d mzero = _mm256_setzero_pd();
    __m256i iall1 = _mm256_set1_epi64x(-1); // iall1 = 0xFF...F
    __m256d dall1 = _mm256_castsi256_pd(iall1);
#define _CMP_GT_OS    0x0e /* Greater-than (ordered, signaling)  */

    __m256d dot_nl = dotproduct(n, l_copy);
    __m256d max = _mm256_cmp_pd(dot_nl, mzero, _CMP_GT_OS); // n2 <= 0.0

    *diffuse = _mm256_and_pd(max, dot_nl); // *diffuse = MAX(0, dot_product(n, l_copy));

    __m256d dot_rd = dotproduct(r, d_copy);
    __m256d max0rd = _mm256_cmp_pd(dot_rd, mzero, _CMP_GT_OS);
    __m256d powtarget = _mm256_and_pd(max0rd, dot_rd);

    // phong_pow should only be the value of 5 or 30
    // need to think how to let pow work in avx
    if (5 == phong_pow) {
        __m256d powtar2 = _mm256_mul_pd(powtar, powtar);
        __m256d powtar4 = _mm256_mul_pd(powtar2, powtar2);
        __m256d powtar5 = _mm256_mul_pd(powtar4, powtar);
        *specular = powtar5;
    } else if (30 == phong_pow) {
        __m256d powtar2 = _mm256_mul_pd(powtar, powtar);
        __m256d powtar4 = _mm256_mul_pd(powtar2, powtar2);
        __m256d powtar8 = _mm256_mul_pd(powtar4, powtar4);
        __m256d powtar16 = _mm256_mul_pd(powtar8, powtar8);
        __m256d powtar24 = _mm256_mul_pd(powtar16, powtar8);
        __m256d powtar6 = _mm256_mul_pd(powtar4, powtar2);
        __m256d powtar30 = _mm256_mul_pd(powtar24, powtar6);
        *specular = powtar30;
    }
}

/* @param r direction of reflected ray
 * @param d direction of primary ray into intersection
 * @param n surface normal at intersection
 */
static void reflection(point3v r, const point3v d, const point3v n)
{
    __m256d dot_dn = dot_product(d, n);
    __m256d tmp = _mm256_set1_pd(-2);
    dot_dn = _mm256_mul_pd(tmp, dot_dn); // -2.0 * dot_product(d,n)
    multiply_vector(n, dot_dn, r);
    add_vector(r, d, r);
}

/* reference: https://www.opengl.org/sdk/docs/man/html/refract.xhtml */
static void refraction(point3v *t, const point3v *I, const point3v *N,
                       double n1, double n2)
{
    __m256d n2v = _mm256_set1_pd(n2);
    __m256d eta = _mm256_set1_pd(n1/n2);
    __m256d dot_NI = dot_product(N, I);
    __m256d k = _mm256_set_pd(1);
    __m256d eta2 = _mm256_mul_pd(eta, eta); // eta2 = eta * eta
    __m256d dot_NI2 = _mm256_mul_pd(dot_NI, dot_NI); // dot_NI2 = dot_NI * dot_NI
    dot_NI2 = _mm256_sub_pd(k, dot_NI2); // dot_NI2 = 1 - dot_NI * dot_NI
    eta2 = _mm256_mul_pd(eta2, dot_NI2); // eta2 = eta * eta * ( 1 - dot_NI * dot_NI)
    k = _mm256_sub_pd(k, eta2); // k = 1 - eta * eta * ( 1 - dot_NI * dot_NI)
#define _CMP_LT_OS    0x01 /* Less-than (ordered, signaling)  */
#define _CMP_LE_OS    0x02 /* Less-than-or-equal (ordered, signaling)  */
    __m256d mzero = _mm256_setzero_pd();
    __m256i iall1 = _mm256_set1_epi64x(-1); // iall1 = 0xFF...F
    __m256d dall1 = _mm256_castsi256_pd(iall1);

    __m256d if1 = _mm256_cmp_pd(k, mzero, _CMP_LT_OS); // k < 0.0
    __m256d if2 = _mm256_cmp_pd(n2v, mzero, _CMP_LE_OS); // n2 <= 0.0
    __m256d ifa = _mm256_or_pd(k, n2v); // if (k < 0.0 || n2 <= 0.0)
    __m256d notifa = _mm256_xor_pd(ifa, dall1);

    point3v tmp;
    multiply_vector(I, eta, t);
    __m256d midpar = _mm256_mul_pd(eta, dot_NI);
    k = _mm256_sqrt_pd(k); // k = sqrt(k)
    midpar = _mm256_add_pd(midpar, k);
    multiply_vector(N, midpar, tmp);
    subtract_vector(t, tmp, t);

    t->x = _mm256_and_pd(t->x, notifa);
    t->y = _mm256_and_pd(t->y, notifa);
    t->z = _mm256_and_pd(t->z, notifa);
}

/* @param i direction of incoming ray, unit vector
 * @param r direction of refraction ray, unit vector
 * @param normal unit vector
 * @param n1 refraction index
 * @param n2 refraction index
 *
 * reference: http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
 */
static double fresnel(const point3 r, const point3 l,
                      const point3 normal, double n1, double n2)
{
    /* TIR */
    if (length(l) < 0.99)
        return 1.0;
    double cos_theta_i = -dot_product(r, normal);
    double cos_theta_t = -dot_product(l, normal);
    double r_vertical_root = (n1 * cos_theta_i - n2 * cos_theta_t) /
                             (n1 * cos_theta_i + n2 * cos_theta_t);
    double r_parallel_root = (n2 * cos_theta_i - n1 * cos_theta_t) /
                             (n2 * cos_theta_i + n1 * cos_theta_t);
    return (r_vertical_root * r_vertical_root +
            r_parallel_root * r_parallel_root) / 2.0;
}

/* @param t distance */
static intersection ray_hit_object(const point3 e, const point3 d,
                                   double t0, double t1,
                                   const rectangular_node rectangulars,
                                   rectangular_node *hit_rectangular,
                                   const sphere_node spheres,
                                   sphere_node *hit_sphere)
{
    /* set these to not hit */
    *hit_rectangular = NULL;
    *hit_sphere = NULL;

    point3 biased_e;
    multiply_vector(d, t0, biased_e);
    add_vector(biased_e, e, biased_e);

    double nearest = t1;
    intersection result, tmpresult;

    for (rectangular_node rec = rectangulars; rec; rec = rec->next) {
        if (rayRectangularIntersection(biased_e, d, &(rec->element),
                                       &tmpresult, &t1) && (t1 < nearest)) {
            /* hit is closest so far */
            *hit_rectangular = rec;
            nearest = t1;
            result = tmpresult;
        }
    }

    /* check the spheres */
    for (sphere_node sphere = spheres; sphere; sphere = sphere->next) {
        if (raySphereIntersection(biased_e, d, &(sphere->element),
                                  &tmpresult, &t1) && (t1 < nearest)) {
            *hit_sphere = sphere;
            *hit_rectangular = NULL;
            nearest = t1;
            result = tmpresult;
        }
    }

    return result;
}

/* @param d direction of ray
 * @param w basic vectors
 */
static void rayConstruction(point3v *d, const point3v *u, const point3v *v,
                            const point3v *w, unsigned int *i, unsigned int *j,
                            const viewpoint *view, unsigned int width,
                            unsigned int height)
{
    double xmin = -0.0175;
    double ymin = -0.0175;
    double xmax =  0.0175;
    double ymax =  0.0175;
    double focal = 0.05;

    point3v u_tmp, v_tmp, w_tmp, s;
    //double w_s = focal;
    //double u_s = xmin + ((xmax - xmin) * (float) i / (width - 1));
    //double v_s = ymax + ((ymin - ymax) * (float) j / (height - 1));
    __m256d w_s =_mm256_set1_pd(focal);
    double temp[4];
    for(int k=0; k<4; k++) {
        temp[k]=xmin + ((xmax - xmin) * (float) i[k] / (width - 1));
    }
    __m256d u_s =_mm256_loadu_pd(temp);
    for(int k=0; k<4; k++) {
        temp[k]=ymax + ((ymin - ymax) * (float) j[k] / (height - 1));
    }
    __m256d v_s =_mm256_loadu_pd(temp);
    /* s = e + u_s * u + v_s * v + w_s * w */
    m_multiply_vector(u, u_s, &u_tmp);
    m_multiply_vector(v, v_s, &v_tmp);
    m_multiply_vector(w, w_s, &w_tmp);
    point3v vrp;
    COPY_POINT3v(&vrp,view->vrp);
    madd_vector(&vrp, &u_tmp, &s);
    madd_vector(&s, &v_tmp, &s);
    madd_vector(&s, &w_tmp, &s);

    /* p(t) = e + td = e + t(s - e) */
    msubtract_vector(&s, &vrp, d);
    mnormalize(d);
}

static void calculateBasisVectors(point3v *u, point3v *v, point3v *w,
                                  const viewpoint *view)
{
    /* w  */
    COPY_POINT3v(w, view->vpn);
    mnormalize(w);

    /* u = (t x w) / (|t x w|) */
    point3v mvup;
    COPY_POINT3v(&mvup, view->vup);
    mcross_product(w, &mvup, u);
    mnormalize(u);

    /* v = w x u */
    mcross_product(u, w, v);

    mnormalize(v);
}

/* @brief protect color value overflow */
static void protect_color_overflow(color c)
{
    for (int i = 0; i < 3; i++)
        if (c[i] > 1.0) c[i] = 1.0;
}

static unsigned int ray_color(const point3 e, double t,
                              const point3 d,
                              idx_stack *stk,
                              const rectangular_node rectangulars,
                              const sphere_node spheres,
                              const light_node lights,
                              color object_color, int bounces_left)
{
    rectangular_node hit_rec = NULL, light_hit_rec = NULL;
    sphere_node hit_sphere = NULL, light_hit_sphere = NULL;
    double diffuse, specular;
    point3 l, _l, r, rr;
    object_fill fill;

    color reflection_part;
    color refraction_part;
    /* might be a reflection ray, so check how many times we've bounced */
    if (bounces_left == 0) {
        SET_COLOR(object_color, 0.0, 0.0, 0.0);
        return 0;
    }

    /* check for intersection with a sphere or a rectangular */
    intersection ip= ray_hit_object(e, d, t, MAX_DISTANCE, rectangulars,
                                    &hit_rec, spheres, &hit_sphere);
    if (!hit_rec && !hit_sphere)
        return 0;

    /* pick the fill of the object that was hit */
    fill = hit_rec ?
           hit_rec->element.rectangular_fill :
           hit_sphere->element.sphere_fill;

    void *hit_obj = hit_rec ? (void *) hit_rec : (void *) hit_sphere;

    /* assume it is a shadow */
    SET_COLOR(object_color, 0.0, 0.0, 0.0);

    for (light_node light = lights; light; light = light->next) {
        /* calculate the intersection vector pointing at the light */
        subtract_vector(ip.point, light->element.position, l);
        multiply_vector(l, -1, _l);
        normalize(_l);
        /* check for intersection with an object. use ignore_me
         * because we don't care about this normal
        */
        ray_hit_object(ip.point, _l, MIN_DISTANCE, length(l),
                       rectangulars, &light_hit_rec,
                       spheres, &light_hit_sphere);
        /* the light was not block by itself(lit object) */
        if (light_hit_rec || light_hit_sphere)
            continue;

        compute_specular_diffuse(&diffuse, &specular, d, l,
                                 ip.normal, fill.phong_power);

        localColor(object_color, light->element.light_color,
                   diffuse, specular, &fill);
    }

    reflection(r, d, ip.normal);
    double idx = idx_stack_top(stk).idx, idx_pass = fill.index_of_refraction;
    if (idx_stack_top(stk).obj == hit_obj) {
        idx_stack_pop(stk);
        idx_pass = idx_stack_top(stk).idx;
    } else {
        idx_stack_element e = { .obj = hit_obj,
                                .idx = fill.index_of_refraction
                              };
        idx_stack_push(stk, e);
    }

    refraction(rr, d, ip.normal, idx, idx_pass);
    double R = (fill.T > 0.1) ?
               fresnel(d, rr, ip.normal, idx, idx_pass) :
               1.0;

    /* totalColor = localColor +
                    mix((1-fill.Kd) * fill.R * reflection, T * refraction, R)
     */
    if (fill.R > 0) {
        /* if we hit something, add the color */
        int old_top = stk->top;
        if (ray_color(ip.point, MIN_DISTANCE, r, stk, rectangulars, spheres,
                      lights, reflection_part,
                      bounces_left - 1)) {
            multiply_vector(reflection_part, R * (1.0 - fill.Kd) * fill.R,
                            reflection_part);
            add_vector(object_color, reflection_part,
                       object_color);
        }
        stk->top = old_top;
    }
    /* calculate refraction ray */
    if ((length(rr) > 0.0) && (fill.T > 0.0) &&
            (fill.index_of_refraction > 0.0)) {
        normalize(rr);
        if (ray_color(ip.point, MIN_DISTANCE, rr, stk,rectangulars, spheres,
                      lights, refraction_part,
                      bounces_left - 1)) {
            multiply_vector(refraction_part, (1 - R) * fill.T,
                            refraction_part);
            add_vector(object_color, refraction_part,
                       object_color);
        }
    }

    protect_color_overflow(object_color);
    return 1;
}


/* @param background_color this is not ambient light */
void raytracing(uint8_t *pixels, color background_color,
                rectangular_node rectangulars, sphere_node spheres,
                light_node lights, const viewpoint *view,
                int width, int height)
{
    point3v u, v, w, d;
    color object_color[4];//= { 0.0, 0.0, 0.0 };
    for(int i=0; i<4; i++) {
        object_color[i][0]=0.0;
        object_color[i][1]=0.0;
        object_color[i][2]=0.0;
    }
    //rgb object_rgb;
    /* calculate u, v, w */
    calculateBasisVectors(&u, &v, &w, view);



    idx_stack stk[4];
    unsigned int i4[4],j4[4];
    int factor = sqrt(SAMPLES);
//    #pragma omp parallel for num_threads (2) private(stk,object_color,d)
    for (int j = 0; j < height; j+=4) {
        for (int i = 0; i < width; i++) {
            double r[4] , g[4] , b[4];
            for(int ii=0; ii<4; ii++) {
                r[ii]=0.0;
                g[ii]=0.0;
                b[ii]=0.0;
            }
            /* MSAA */
            for (int s = 0; s < SAMPLES; s++) {
                idx_stack_init(&stk[0]);
                idx_stack_init(&stk[1]);
                idx_stack_init(&stk[2]);
                idx_stack_init(&stk[3]);
                for(int k=0; k<4; k++) {
                    i4[k]= i * factor + s / factor;
                    j4[k]= (j+k) * factor + s % factor;
                }
                rayConstruction(&d, &u, &v, &w,
                                i4,
                                j4,
                                view,
                                width * factor, height * factor);
                //point3v vrp;
                //COPY_POINT3v(vrp,view->vrp);
                point3 dp[4];
                double x[4];
                double y[4];
                double z[4];
                _mm256_storeu_pd(x,d.x);
                _mm256_storeu_pd(y,d.y);
                _mm256_storeu_pd(z,d.z);
                for(int ii=0; ii<4; ii++) {
                    dp[ii][0]=x[ii];
                    dp[ii][1]=y[ii];
                    dp[ii][2]=z[ii];
                }
                for(int k=0; k<4; k++) {
                    if (ray_color(view->vrp, 0.0, dp[k], &(stk[k]), rectangulars, spheres,
                                  lights, object_color[k],
                                  MAX_REFLECTION_BOUNCES)) {
                        r[k] += object_color[k][0];
                        g[k] += object_color[k][1];
                        b[k] += object_color[k][2];
                    } else {
                        r[k] += background_color[0];
                        g[k] += background_color[1];
                        b[k] += background_color[2];
                    }
                    pixels[((i + ((j+k) * width)) * 3) + 0] = r[k] * 255 / SAMPLES;
                    pixels[((i + ((j+k) * width)) * 3) + 1] = g[k] * 255 / SAMPLES;
                    pixels[((i + ((j+k) * width)) * 3) + 2] = b[k] * 255 / SAMPLES;
                    if(i==width/2&&j==200&&k==0)
                        printf("%lf %lf %lf\n",r[k],g[k],b[k]);
                }
            }
        }
    }
}

