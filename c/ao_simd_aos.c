#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <time.h>

#define WIDTH        256
#define HEIGHT       256
#define NSUBSAMPLES  2
#define NAO_SAMPLES  8

typedef __attribute__ ((__aligned__(16))) __m128 vec;

typedef struct __attribute__ ((__aligned__(16)))
{
    vec    p;
    vec    n;
    vec    t;
    int    hit; 
} Isect;

typedef struct __attribute__ ((__aligned__(16)))
{
    vec    center;
    vec    radius;

} Sphere;

typedef struct __attribute__ ((__aligned__(16)))
{
    vec    p;
    vec    n;

} Plane;

typedef struct __attribute__ ((__aligned__(16)))
{
    vec    org;
    vec    dir;
} Ray;

Sphere spheres[3];
Plane  plane;

#define SIGNMASK _mm_castsi128_ps(_mm_set1_epi32(0x80000000))
#define ZEROS _mm_setzero_ps()

#define _MM_SHUFFLE_(w, z, y, x) ((w<<6)|(z<<4)|(y<<2)|x)
#define _MM_NEGATE_(v) (_mm_xor_ps(v, SIGNMASK))
#define _MM_ABSVAL_(v) (_mm_andnot_ps(SIGNMASK, v))

static vec vdot(vec v0, vec v1)
{
    /*
    vec v = _mm_mul_ps(v0, v1);
    return _mm_add_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)), 
                      _mm_add_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)), _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))));
                      */
    return _mm_dp_ps(v0, v1, 0xff);
}

static vec vcross(vec v0, vec v1)
{
    return _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 1, 0, 2))),
                      _mm_mul_ps(_mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 0, 2, 1))));
}

static vec vnormalize(vec c)
{
    vec length = _mm_sqrt_ps(vdot(c, c));
    
    if (_mm_movemask_ps(_mm_cmpgt_ps(length, _mm_set1_ps(1.0e-17)))) {
        return _mm_div_ps(c, length);
    } else {
        return c;
    }

}

void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs = _mm_sub_ps(ray->org, sphere->center);
    
    vec B = vdot(rs, ray->dir);
    vec C = _mm_sub_ps(vdot(rs, rs), _mm_mul_ps(sphere->radius, sphere->radius));
    vec D = _mm_sub_ps(_mm_mul_ps(B, B), C);
    
    if (_mm_movemask_ps(_mm_cmpgt_ps(D, ZEROS))) {
        vec t = _mm_sub_ps(_MM_NEGATE_(B), _mm_sqrt_ps(D));
        
        if (_mm_movemask_ps(_mm_and_ps(_mm_cmpgt_ps(t, ZEROS), _mm_cmplt_ps(t, isect->t)))) {
            isect->t = t;
            isect->hit = 1;
            
            isect->p = _mm_add_ps(ray->org, _mm_mul_ps(ray->dir, t));
            isect->n = _mm_sub_ps(isect->p, sphere->center);
            
            isect->n = vnormalize(isect->n);
        }
    }
}

void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    vec d = _MM_NEGATE_(vdot(plane->p, plane->n));
    vec v = vdot(ray->dir, plane->n);
    
    if (_mm_movemask_ps(_mm_cmplt_ps(_MM_ABSVAL_(v), _mm_set1_ps(1.0e-17))))  return;
    
    vec t = _mm_div_ps(_MM_NEGATE_(_mm_add_ps(vdot(ray->org, plane->n), d)), v);
    
    if (_mm_movemask_ps(_mm_and_ps(_mm_cmpgt_ps(t, ZEROS), _mm_cmplt_ps(t, isect->t)))) {
        isect->t = t;
        isect->hit = 1;
        isect->p = _mm_add_ps(ray->org, _mm_mul_ps(ray->dir, t));
        isect->n = plane->n;
    }
}

void
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    
    vec mask = _mm_cmplt_ps(_MM_ABSVAL_(n), _mm_set1_ps(0.6));
    vec xmask = _mm_and_ps(_mm_set_ps(1.0, 1.0, 1.0, 0.0),
                           _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(0, 0, 0, 0)));
    vec ymask = _mm_and_ps(_mm_set_ps(1.0, 1.0, 0.0, 0.0),
                           _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(1, 1, 1, 1)));
    vec yzmask = _mm_andnot_ps(_mm_or_ps(_mm_shuffle_ps(mask, mask, _MM_SHUFFLE(1, 1, 1, 1)),
                                         _mm_shuffle_ps(mask, mask, _MM_SHUFFLE(2, 2, 2, 2))),
                               _mm_set_ps(0.0, 0.0, 0.0, 1.0));
    mask = _mm_or_ps(_mm_andnot_ps(ymask, _mm_andnot_ps(xmask, mask)), yzmask);
    basis[1] = _mm_and_ps(_mm_set_ps(0.0, 1.0, 1.0, 1.0), mask);
    
    basis[0] = vcross(basis[1], basis[2]);
    basis[0] = vnormalize(basis[0]);

    basis[1] = vcross(basis[2], basis[0]);
    basis[1] = vnormalize(basis[1]);
}


void ambient_occlusion(vec *col, const Isect *isect)
{
    int    i, j;
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    vec    eps = _mm_set1_ps(0.0001);
    
    vec p = _mm_add_ps(isect->p, _mm_mul_ps(eps, isect->n));
    
    vec basis[3];
    orthoBasis(basis, isect->n);
    
    float occlusion = 0.0;
    
    for (j = 0; j < ntheta; j++) {
        for (i = 0; i < nphi; i++) {
            
            float theta = sqrt(drand48());
            float phi   = 2.0 * M_PI * drand48();
            
            float x = cos(phi) * theta;
            float y = sin(phi) * theta;
            float z = sqrt(1.0 - theta * theta);
            vec xvec = _mm_set1_ps(x);
            vec yvec = _mm_set1_ps(y);
            vec zvec = _mm_set1_ps(z);
            
            // local -> global
            vec rv = _mm_add_ps(_mm_mul_ps(xvec, basis[0]),
                                _mm_add_ps(_mm_mul_ps(yvec, basis[1]),
                                           _mm_mul_ps(zvec, basis[2])));
            
            Ray ray;
            
            ray.org = p;
            ray.dir = rv;
            
            Isect occIsect;
            occIsect.t   = _mm_set1_ps(1.0e+17);
            occIsect.hit = 0;
            
            ray_sphere_intersect(&occIsect, &ray, &spheres[0]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[1]); 
            ray_sphere_intersect(&occIsect, &ray, &spheres[2]); 
            ray_plane_intersect (&occIsect, &ray, &plane); 
            
            if (occIsect.hit) occlusion += 1.0;
            
        }
    }
    
    occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
    
    *col = _mm_set1_ps(occlusion);
}

static inline uint32_t clamp(vec _v)
{
    __m128i v = _mm_cvttps_epi32(_mm_mul_ps(_v, _mm_set1_ps(255.5)));
    v = _mm_packus_epi16(_mm_packs_epi32(v, _mm_setzero_si128()), _mm_setzero_si128());
    return _mm_extract_epi32(v, 0);
}


void
render(unsigned char *img, int w, int h, int nsubsamples)
{
    int x, y;
    int u, v;
    
    /*
    float *fimg = (float *)malloc(sizeof(float) * w * h * 3);
    memset((void *)fimg, 0, sizeof(float) * w * h * 3);
    */
    
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {

            vec fimg = ZEROS;
            
            for (v = 0; v < nsubsamples; v++) {
                for (u = 0; u < nsubsamples; u++) {
                    
                    float px = (x + (u / (float)nsubsamples) - (w / 2.0)) / (w / 2.0);
                    float py = -(y + (v / (float)nsubsamples) - (h / 2.0)) / (h / 2.0);
                    
                    Ray ray;
                    ray.org = ZEROS;
                    ray.dir = _mm_set_ps(0.0, -1.0, py, px);
                    ray.dir = vnormalize(ray.dir);
                    
                    Isect isect;
                    isect.t   = _mm_set1_ps(1.0e+17);
                    isect.hit = 0;
                    
                    ray_sphere_intersect(&isect, &ray, &spheres[0]);
                    ray_sphere_intersect(&isect, &ray, &spheres[1]);
                    ray_sphere_intersect(&isect, &ray, &spheres[2]);
                    ray_plane_intersect (&isect, &ray, &plane);
                    
                    if (isect.hit) {
                        vec col;
                        ambient_occlusion(&col, &isect);
                        fimg = _mm_add_ps(fimg, col);
                    }
                    
                }
            }
            
            fimg = _mm_div_ps(fimg, _mm_set1_ps((float)(nsubsamples * nsubsamples)));
            
            uint32_t iimg = clamp(fimg);
            memcpy((void *)&img[3 * (y * w + x)], &iimg, 3);

        }
    }
}

void
init_scene()
{
    spheres[0].center = _mm_set_ps(0.0, -3.5, 0.0, -2.0);
    spheres[0].radius = _mm_set1_ps(0.5);
    
    spheres[1].center = _mm_set_ps(0.0, -3.0, 0.0, -0.5);
    spheres[1].radius = _mm_set1_ps(0.5);            
    
    spheres[2].center = _mm_set_ps(0.0, -2.2, 0.0, 1.0);
    spheres[2].radius = _mm_set1_ps(0.5);
    
    plane.p = _mm_set_ps(0.0, 0.0, -0.5, 0.0);
    plane.n = _mm_set_ps(0.0, 0.0, 1.0, 0.0);

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

    render(img, WIDTH, HEIGHT, NSUBSAMPLES);

    saveppm("ao.ppm", WIDTH, HEIGHT, img); 

    return 0;
}
