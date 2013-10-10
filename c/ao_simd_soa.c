#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include "lib/sse_mathfun.h"

#define WIDTH        256
#define HEIGHT       256
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8

typedef struct _vec
{
    float x;
    float y;
    float z;
} vec;


typedef struct _Isect
{
    float t;
    vec    p;
    vec    n;
    int    hit; 
} Isect;

typedef struct _Sphere
{
    vec    center;
    float radius;

} Sphere;

typedef struct _Plane
{
    vec    p;
    vec    n;

} Plane;

typedef struct _Ray
{
    vec    org;
    vec    dir;
} Ray;

Sphere spheres[3];
Plane  plane;

#define SIGNMASK _mm_castsi128_ps(_mm_set1_epi32(0x80000000))
#define _MM_NEGATE_(v) (_mm_xor_ps(v, SIGNMASK))
#define _MM_ABSVAL_(v) (_mm_andnot_ps(SIGNMASK, v))

static float vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static void vcross(vec *c, vec v0, vec v1)
{
    
    c->x = v0.y * v1.z - v0.z * v1.y;
    c->y = v0.z * v1.x - v0.x * v1.z;
    c->z = v0.x * v1.y - v0.y * v1.x;
}

static void vnormalize(vec *c)
{
    float length = sqrt(vdot((*c), (*c)));

    if (fabs(length) > 1.0e-17) {
        c->x /= length;
        c->y /= length;
        c->z /= length;
    }
}

void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs;

    rs.x = ray->org.x - sphere->center.x;
    rs.y = ray->org.y - sphere->center.y;
    rs.z = ray->org.z - sphere->center.z;

    float B = vdot(rs, ray->dir);
    float C = vdot(rs, rs) - sphere->radius * sphere->radius;
    float D = B * B - C;

    if (D > 0.0) {
        float t = -B - sqrt(D);
        
        if ((t > 0.0) && (t < isect->t)) {
            isect->t = t;
            isect->hit = 1;
            
            isect->p.x = ray->org.x + ray->dir.x * t;
            isect->p.y = ray->org.y + ray->dir.y * t;
            isect->p.z = ray->org.z + ray->dir.z * t;

            isect->n.x = isect->p.x - sphere->center.x;
            isect->n.y = isect->p.y - sphere->center.y;
            isect->n.z = isect->p.z - sphere->center.z;

            vnormalize(&(isect->n));
        }
    }
}

static inline void
ray_sphere_intersect_simd(__m128 *t, __m128 *hit,
                          __m128 *px, __m128 *py, __m128 *pz,
                          __m128 *nx, __m128 *ny, __m128 *nz,
                          const __m128 dirx, const __m128 diry, const __m128 dirz,
                          const __m128 orgx, const __m128 orgy, const __m128 orgz,
                          const Sphere *sphere)
{
    __m128 rsx = _mm_sub_ps(orgx, _mm_set1_ps(sphere->center.x));
    __m128 rsy = _mm_sub_ps(orgy, _mm_set1_ps(sphere->center.y));
    __m128 rsz = _mm_sub_ps(orgz, _mm_set1_ps(sphere->center.z));
    
    __m128 B = _mm_add_ps(_mm_mul_ps(rsx, dirx), 
                          _mm_add_ps(_mm_mul_ps(rsy, diry), _mm_mul_ps(rsz, dirz)));
    __m128 C = _mm_sub_ps(_mm_add_ps(_mm_mul_ps(rsx, rsx), 
                                     _mm_add_ps(_mm_mul_ps(rsy, rsy), _mm_mul_ps(rsz, rsz))),
                          _mm_set1_ps(sphere->radius * sphere->radius));
    __m128 D = _mm_sub_ps(_mm_mul_ps(B, B), C);
    
    __m128 cond1 = _mm_cmpgt_ps(D, _mm_set1_ps(0.0));
    if (_mm_movemask_ps(cond1)) {
        __m128 t2 = _mm_sub_ps(_MM_NEGATE_(B), _mm_sqrt_ps(D));
        __m128 cond2 = _mm_and_ps(_mm_cmpgt_ps(t2, _mm_set1_ps(0.0)), _mm_cmplt_ps(t2, *t));
        if (_mm_movemask_ps(cond2)) {
            *t = _mm_or_ps(_mm_and_ps(cond2, t2), _mm_andnot_ps(cond2, *t));
            *hit = _mm_or_ps(cond2, *hit);
            
            *px = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgx, _mm_mul_ps(dirx, *t))), 
                            _mm_andnot_ps(cond2, *px));
            *py = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgy, _mm_mul_ps(diry, *t))), 
                            _mm_andnot_ps(cond2, *py));
            *pz = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgz, _mm_mul_ps(dirz, *t))), 
                            _mm_andnot_ps(cond2, *pz));

            *nx = _mm_or_ps(_mm_and_ps(cond2, _mm_sub_ps(*px, _mm_set1_ps(sphere->center.x))), 
                            _mm_andnot_ps(cond2, *nx));
            *ny = _mm_or_ps(_mm_and_ps(cond2, _mm_sub_ps(*py, _mm_set1_ps(sphere->center.y))), 
                            _mm_andnot_ps(cond2, *ny));
            *nz = _mm_or_ps(_mm_and_ps(cond2, _mm_sub_ps(*pz, _mm_set1_ps(sphere->center.z))), 
                            _mm_andnot_ps(cond2, *nz));
    
            __m128 lengths = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(*nx, *nx),
                                                    _mm_add_ps(_mm_mul_ps(*ny, *ny), 
                                                               _mm_mul_ps(*nz, *nz))));
            __m128 cond3 = _mm_cmpgt_ps(_MM_ABSVAL_(lengths), _mm_set1_ps(1.0e-17));
            *nx = _mm_or_ps(_mm_and_ps(cond3, _mm_div_ps(*nx, lengths)), _mm_andnot_ps(cond3, *nx));
            *ny = _mm_or_ps(_mm_and_ps(cond3, _mm_div_ps(*ny, lengths)), _mm_andnot_ps(cond3, *ny));
            *nz = _mm_or_ps(_mm_and_ps(cond3, _mm_div_ps(*nz, lengths)), _mm_andnot_ps(cond3, *nz));
        }
    }
}


void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    float d = -vdot(plane->p, plane->n);
    float v = vdot(ray->dir, plane->n);
    
    if (fabs(v) < 1.0e-17) return;
    
    float t = -(vdot(ray->org, plane->n) + d) / v;
    
    if ((t > 0.0) && (t < isect->t)) {
        isect->t = t;
        isect->hit = 1;
        
        isect->p.x = ray->org.x + ray->dir.x * t;
        isect->p.y = ray->org.y + ray->dir.y * t;
        isect->p.z = ray->org.z + ray->dir.z * t;

        isect->n = plane->n;
    }
}

static inline void
ray_plane_intersect_simd(__m128 *t, __m128 *hit,
                         __m128 *px, __m128 *py, __m128 *pz,
                         __m128 *nx, __m128 *ny, __m128 *nz,
                         const __m128 dirx, const __m128 diry, const __m128 dirz,
                         const __m128 orgx, const __m128 orgy, const __m128 orgz,
                         const Plane *plane)
{
    __m128 d = _MM_NEGATE_(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(plane->p.x), _mm_set1_ps(plane->n.x)), 
                                      _mm_add_ps(_mm_mul_ps(_mm_set1_ps(plane->p.y), 
                                                            _mm_set1_ps(plane->n.y)), 
                                                 _mm_mul_ps(_mm_set1_ps(plane->p.z), 
                                                            _mm_set1_ps(plane->n.z)))));
    __m128 v = _mm_add_ps(_mm_mul_ps(dirx, _mm_set1_ps(plane->n.x)), 
                          _mm_add_ps(_mm_mul_ps(diry, _mm_set1_ps(plane->n.y)), 
                                     _mm_mul_ps(dirz, _mm_set1_ps(plane->n.z))));
    
    __m128 cond1 = _mm_cmpgt_ps(_MM_ABSVAL_(v), _mm_set1_ps(1.0e-17));
    __m128 dp = _mm_add_ps(_mm_mul_ps(orgx, _mm_set1_ps(plane->n.x)), 
                           _mm_add_ps(_mm_mul_ps(orgy, _mm_set1_ps(plane->n.y)), 
                                      _mm_mul_ps(orgz, _mm_set1_ps(plane->n.z))));
    __m128 t2 = _mm_and_ps(cond1, _mm_div_ps(_MM_NEGATE_(_mm_add_ps(dp, d)), v));
    __m128 cond2 = _mm_and_ps(_mm_cmpgt_ps(t2, _mm_set1_ps(0.0)), _mm_cmplt_ps(t2, *t));
    if (_mm_movemask_ps(cond2)) {
        *t = _mm_or_ps(_mm_and_ps(cond2, t2), _mm_andnot_ps(cond2, *t));
        *hit = _mm_or_ps(cond2, *hit);
    
        *px = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgx, _mm_mul_ps(dirx, *t))), 
                        _mm_andnot_ps(cond2, *px));
        *py = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgy, _mm_mul_ps(diry, *t))), 
                        _mm_andnot_ps(cond2, *py));
        *pz = _mm_or_ps(_mm_and_ps(cond2, _mm_add_ps(orgz, _mm_mul_ps(dirz, *t))), 
                        _mm_andnot_ps(cond2, *pz));

        *nx = _mm_or_ps(_mm_and_ps(cond2, _mm_set1_ps(plane->n.x)), 
                        _mm_andnot_ps(cond2, *nx));
        *ny = _mm_or_ps(_mm_and_ps(cond2, _mm_set1_ps(plane->n.y)), 
                        _mm_andnot_ps(cond2, *ny));
        *nz = _mm_or_ps(_mm_and_ps(cond2, _mm_set1_ps(plane->n.z)), 
                        _mm_andnot_ps(cond2, *nz));
    }
}

void
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    vcross(&basis[0], basis[1], basis[2]);
    vnormalize(&basis[0]);

    vcross(&basis[1], basis[2], basis[0]);
    vnormalize(&basis[1]);
}


void ambient_occlusion(vec *col, const Isect *isect)
{
    int    i, j;
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    float eps = 0.0001;
    
    vec p;
    
    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;
    
    vec basis[3];
    orthoBasis(basis, isect->n);
    
    float occlusion = 0.0;
    __m128 occlusionx4 = _mm_set1_ps(0.0);
    
    for (j = 0; j < ntheta; j++) {
        
        float __attribute__ ((__aligned__(16))) rand1[nphi];
        float __attribute__ ((__aligned__(16))) rand2[nphi];
        
        for (i = 0; i < nphi; i++) {
            rand1[i] = drand48();
            rand2[i] = drand48();
        }
            
        assert((nphi % 4) == 0);
        for (i = 0; i < nphi; i += 4) {
            
            __m128 theta = _mm_sqrt_ps(_mm_load_ps(&rand1[i]));
            __m128 phi = _mm_mul_ps(_mm_set1_ps(2.0 * M_PI), _mm_load_ps(&rand2[i]));
            __m128 sinphi;
            __m128 cosphi;
            sincos_ps(phi, &sinphi, &cosphi);
            __m128 x = _mm_mul_ps(cosphi, theta);
            __m128 y = _mm_mul_ps(sinphi, theta);
            __m128 z = _mm_sqrt_ps(_mm_sub_ps(_mm_set1_ps(1.0), _mm_mul_ps(theta, theta)));
            
            // ray
            __m128 dirx = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(basis[0].x)),
                                     _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(basis[1].x)),
                                                _mm_mul_ps(z, _mm_set1_ps(basis[2].x))));
            __m128 diry = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(basis[0].y)),
                                     _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(basis[1].y)),
                                                _mm_mul_ps(z, _mm_set1_ps(basis[2].y))));
            __m128 dirz = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(basis[0].z)),
                                     _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(basis[1].z)),
                                                _mm_mul_ps(z, _mm_set1_ps(basis[2].z))));
            __m128 orgx = _mm_set1_ps(p.x);
            __m128 orgy = _mm_set1_ps(p.y);
            __m128 orgz = _mm_set1_ps(p.z);
            
            // isect
            __m128 t =  _mm_set1_ps(1.0e+17);
            __m128 hit = _mm_set1_ps(0.0);
            __m128 px, py, pz;
            __m128 nx, ny, nz;
            
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[0]);
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[1]);
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[2]);
            ray_plane_intersect_simd (&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &plane);
            
            occlusionx4 = _mm_add_ps(occlusionx4, _mm_and_ps(hit, _mm_set1_ps(1.0f)));
            
        }
    }
    
    float __attribute__ ((__aligned__(16))) occlusionTmp[4];
    _mm_store_ps(occlusionTmp, occlusionx4);
    occlusion = occlusionTmp[0] + occlusionTmp[1] + occlusionTmp[2] + occlusionTmp[3];
    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);

#if DBG
    fprintf(stderr, ".2%f\n", occlusion);
#endif

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

unsigned char
clamp(float f)
{
    
  int i = (int)(f * 255.5);

  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)i;
}


void
render(unsigned char *img, int w, int h, int nsubsamples)
{
    int x, y;
    int u, v;

    float *fimg = (float *)malloc(sizeof(float) * w * h * 3);
    memset((void *)fimg, 0, sizeof(float) * w * h * 3);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {

                for (v = 0; v < nsubsamples; v++) {
                for (u = 0; u < nsubsamples; u++) {
                    
                    float px = (x + (u / (float)nsubsamples) - (w / 2.0)) / (w / 2.0);
                    float py = -(y + (v / (float)nsubsamples) - (h / 2.0)) / (h / 2.0);

                    Ray ray;

                    ray.org.x = 0.0;
                    ray.org.y = 0.0;
                    ray.org.z = 0.0;

                    ray.dir.x = px;
                    ray.dir.y = py;
                    ray.dir.z = -1.0;
                    vnormalize(&(ray.dir));
                    
                    Isect isect;
                    isect.t   = 1.0e+17;
                    isect.hit = 0;
                    
                    ray_sphere_intersect(&isect, &ray, &spheres[0]);
                    ray_sphere_intersect(&isect, &ray, &spheres[1]);
                    ray_sphere_intersect(&isect, &ray, &spheres[2]);
                    ray_plane_intersect (&isect, &ray, &plane);
                    
                    if (isect.hit) {
                        vec col;
#if DBG
                        fprintf(stderr, "%d %d %d %d\t", y, x, v, u);
#endif
                        ambient_occlusion(&col, &isect);
                        
                        fimg[3 * (y * w + x) + 0] += col.x;
                        fimg[3 * (y * w + x) + 1] += col.y;
                        fimg[3 * (y * w + x) + 2] += col.z;
                    }

                }
            }

            fimg[3 * (y * w + x) + 0] /= (float)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 1] /= (float)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 2] /= (float)(nsubsamples * nsubsamples);
                    
            img[3 * (y * w + x) + 0] = clamp(fimg[3 *(y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 *(y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 *(y * w + x) + 2]);
            
        }
    }
}

void
init_scene()
{
    spheres[0].center.x = -2.0;
    spheres[0].center.y =  0.0;
    spheres[0].center.z = -3.5;
    spheres[0].radius = 0.5;
    
    spheres[1].center.x = -0.5;
    spheres[1].center.y =  0.0;
    spheres[1].center.z = -3.0;
    spheres[1].radius = 0.5;
    
    spheres[2].center.x =  1.0;
    spheres[2].center.y =  0.0;
    spheres[2].center.z = -2.2;
    spheres[2].radius = 0.5;

    plane.p.x = 0.0;
    plane.p.y = -0.5;
    plane.p.z = 0.0;

    plane.n.x = 0.0;
    plane.n.y = 1.0;
    plane.n.z = 0.0;

}

void
saveppm(const char *fname, int w, int h, unsigned char *img)
{
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
}

int
main(int argc, char **argv)
{
    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    init_scene();

#ifndef DBG
    clock_t start = clock();
#endif
    render(img, WIDTH, HEIGHT, NSUBSAMPLES);
#ifndef DBG
    clock_t elapsed = clock() - start;
    printf("%.2f sec\n", ((float) elapsed)/CLOCKS_PER_SEC);
#endif

    saveppm("ao.ppm", WIDTH, HEIGHT, img); 

    return 0;
}
