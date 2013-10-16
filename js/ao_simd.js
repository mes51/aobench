var WIDTH  = 256;
var HEIGHT = 256;
var NSUBSAMPLES = 2;
var NAO_SAMPLES = 8;

var spheres;
var plane;

function vdot(v0, v1) 
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}


function vcross(v0, v1) 
{
    return {
        x: v0.y * v1.z - v0.z * v1.y,
        y: v0.z * v1.x - v0.x * v1.z,
        z: v0.x * v1.y - v0.y * v1.x
    };
}


function vnormalize(c) 
{
    var length = Math.sqrt(vdot(c, c));
    if (Math.abs(length) > 1e-17) {
        c.x /= length;
        c.y /= length;
        c.z /= length;
    }
}


function ray_sphere_intersect(isect, ray, sphere) 
{
    var rs = {
        x: ray.org.x - sphere.center.x,
        y: ray.org.y - sphere.center.y,
        z: ray.org.z - sphere.center.z
    };
    
    var B = vdot(rs, ray.dir);
    var C = vdot(rs, rs) - sphere.radius * sphere.radius;
    var D = B * B - C;
    
    if (D > 0) {
        var t = -B - Math.sqrt(D);
        if ((t > 0) && (t < isect.t)) {
            
            isect.t = t;
            isect.hit = 1;
            
            isect.p.x = ray.org.x + ray.dir.x * t;
            isect.p.y = ray.org.y + ray.dir.y * t;
            isect.p.z = ray.org.z + ray.dir.z * t;
            
            isect.n.x = isect.p.x - sphere.center.x;
            isect.n.y = isect.p.y - sphere.center.y;
            isect.n.z = isect.p.z - sphere.center.z;
            
            vnormalize(isect.n);
            
        }
    }
}


function ray_sphere_intersect_simd(isect, dirx, diry, dirz, orgx, orgy, orgz, sphere) 
{
    var rsx = SIMD.sub(orgx, float32x4.splat(sphere.center.x));
    var rsy = SIMD.sub(orgy, float32x4.splat(sphere.center.y));
    var rsz = SIMD.sub(orgz, float32x4.splat(sphere.center.z));
    
    var B = SIMD.add(SIMD.mul(rsx, dirx),
                     SIMD.add(SIMD.mul(rsy, diry), SIMD.mul(rsz, dirz)));
    var C = SIMD.sub(SIMD.add(SIMD.mul(rsx, rsx),
                              SIMD.add(SIMD.mul(rsy, rsy), SIMD.mul(rsz, rsz))),
                     float32x4.splat(sphere.radius * sphere.radius));
    var D = SIMD.sub(SIMD.mul(B, B), C);
    
    var cond1 = SIMD.greaterThan(D, float32x4.zero());
    if (cond1.signMask) {
        var t2 = SIMD.sub(SIMD.neg(B), SIMD.sqrt(D));
        var cond2 = SIMD.and(SIMD.greaterThan(t2, float32x4.zero()), 
                             SIMD.lessThan(t2, isect.t));
        if (cond2.signMask) {
            isect.t = SIMD.bitsToFloat32x4(SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(t2)), 
                                                   SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.t))));
            isect.hit = SIMD.or(cond2, SIMD.bitsToUint32x4(isect.hit));
            
            isect.p.x = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(dirx, isect.t)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.x)))); 
            isect.p.y = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(diry, isect.t)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.y)))); 
            isect.p.z = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(dirz, isect.t)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.z)))); 
                        
            isect.n.x = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.sub(isect.p.x, float32x4.splat(sphere.center.x)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.x)))); 
            isect.n.y = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.sub(isect.p.y, float32x4.splat(sphere.center.y)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.y)))); 
            isect.n.z = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.sub(isect.p.z, float32x4.splat(sphere.center.z)))),
                        SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.z)))); 
                        
            var lengths = SIMD.sqrt(SIMD.add(SIMD.mul(isect.n.x, isect.n.x),
                                             SIMD.add(SIMD.mul(isect.n.y, isect.n.y),
                                                      SIMD.mul(isect.n.z, isect.n.z))));
            var cond3 = SIMD.greaterThan(SIMD.abs(lengths), float32x4.splat(1e-17));
            isect.n.x = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond3, SIMD.bitsToUint32x4(SIMD.div(isect.n.x, lengths))),
                        SIMD.and(SIMD.negu32(cond3), SIMD.bitsToUint32x4(isect.n.x))));
            isect.n.y = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond3, SIMD.bitsToUint32x4(SIMD.div(isect.n.y, lengths))),
                        SIMD.and(SIMD.negu32(cond3), SIMD.bitsToUint32x4(isect.n.y))));
            isect.n.z = SIMD.bitsToFloat32x4(
                SIMD.or(SIMD.and(cond3, SIMD.bitsToUint32x4(SIMD.div(isect.n.z, lengths))),
                        SIMD.and(SIMD.negu32(cond3), SIMD.bitsToUint32x4(isect.n.z))));
        }
    }
}


function ray_plane_intersect(isect, ray, plane) 
{
    var d = -vdot(plane.p, plane.n);
    var v = vdot(ray.dir, plane.n);
    
    if (Math.abs(v) < 1e-17) return;
    
    var t = -(vdot(ray.org, plane.n) + d) / v;
    
    if ((t > 0) && (t < isect.t)) {
        isect.t = t;
        isect.hit = 1;
        isect.p.x = ray.org.x + ray.dir.x * t;
        isect.p.y = ray.org.y + ray.dir.y * t;
        isect.p.z = ray.org.z + ray.dir.z * t;
        isect.n = plane.n;
    }
}


function ray_plane_intersect_simd(isect, dirx, diry, dirz, orgx, orgy, orgz, plane) 
{
    var d = SIMD.neg(SIMD.add(SIMD.mul(float32x4.splat(plane.p.x), float32x4.splat(plane.n.x)),
                              SIMD.add(SIMD.mul(float32x4.splat(plane.p.y), float32x4.splat(plane.n.y)),
                                       SIMD.mul(float32x4.splat(plane.p.z), float32x4.splat(plane.n.z)))));
    var v = SIMD.add(SIMD.mul(dirx, float32x4.splat(plane.n.x)),
                     SIMD.add(SIMD.mul(diry, float32x4.splat(plane.n.y)),
                              SIMD.mul(dirz, float32x4.splat(plane.n.z))));
    
    var cond1 = SIMD.greaterThan(SIMD.abs(v), float32x4.splat(1e-17));
    var dp = SIMD.add(SIMD.mul(orgx, float32x4.splat(plane.n.x)),
                      SIMD.add(SIMD.mul(orgy, float32x4.splat(plane.n.y)),
                               SIMD.mul(orgz, float32x4.splat(plane.n.z))));
    var t2 = SIMD.bitsToFloat32x4(SIMD.and(cond1, SIMD.bitsToUint32x4(SIMD.div(SIMD.neg(SIMD.add(dp, d)), v))));                              
    var cond2 = SIMD.and(SIMD.greaterThan(t2, float32x4.zero()), SIMD.lessThan(t2, isect.t));
    if (cond2.signMask) {
        isect.t = SIMD.bitsToFloat32x4(SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(t2)),
                                               SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.t))));
        isect.hit = SIMD.or(cond2, SIMD.bitsToUint32x4(isect.hit));
        isect.p.x = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(dirx, isect.t)))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.x)))); 
        isect.p.y = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(diry, isect.t)))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.y)))); 
        isect.p.z = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(SIMD.add(orgx, SIMD.mul(dirz, isect.t)))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.p.z)))); 
        
        isect.n.x = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(float32x4.splat(plane.n.x))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.x)))); 
        isect.n.y = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(float32x4.splat(plane.n.y))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.y)))); 
        isect.n.z = SIMD.bitsToFloat32x4(
            SIMD.or(SIMD.and(cond2, SIMD.bitsToUint32x4(float32x4.splat(plane.n.z))),
                    SIMD.and(SIMD.negu32(cond2), SIMD.bitsToUint32x4(isect.n.z)))); 
    }
}


function orthoBasis(basis, n) 
{
    basis[2] = n;
    basis[1] = { x: 0, y: 0, z: 0 };
    
    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }
    
    basis[0] = vcross(basis[1], basis[2]);
    vnormalize(basis[0]);
    
    basis[1] = vcross(basis[2], basis[0]);
    vnormalize(basis[1]);
}


function ambient_occlusion(col, isect) 
{
    var i, j;
    var ntheta = NAO_SAMPLES;
    var nphi = NAO_SAMPLES;
    var eps = 0.0001;
    
    var p = {
        x: isect.p.x + eps * isect.n.x,
        y: isect.p.y + eps * isect.n.y,
        z: isect.p.z + eps * isect.n.z
    };
    
    var basis = new Array({}, {}, {});
    orthoBasis(basis, isect.n);
    
    var occlusion = 0;
    var occlusionx4 = float32x4.zero();
    
    for (j = 0; j < ntheta; j++) {
        
        if ((nphi % 4) != 0) {
            throw "update code to account for stripmining remainder loop"
        }
        
        for (i = 0; i < nphi; i += 4) {
           
            /* TODO: implement sin/cos library to work with sse */
            var theta = SIMD.sqrt(float32x4(Math.random(), Math.random(), Math.random(), Math.random()));
            var phi0 = 2 * Math.PI * Math.random();
            var phi1 = 2 * Math.PI * Math.random();
            var phi2 = 2 * Math.PI * Math.random();
            var phi3 = 2 * Math.PI * Math.random();
            var sinphi = float32x4(Math.sin(phi0), Math.sin(phi1), Math.sin(phi2), Math.sin(phi3));
            var cosphi = float32x4(Math.cos(phi0), Math.cos(phi1), Math.cos(phi2), Math.cos(phi3));
            
            var x = SIMD.mul(cosphi, theta);
            var y = SIMD.mul(sinphi, theta);
            var z = SIMD.sqrt(SIMD.sub(float32x4.splat(1), SIMD.mul(theta, theta)));
            
            var dirx = SIMD.add(SIMD.mul(x, float32x4.splat(basis[0].x)),
                                SIMD.add(SIMD.mul(y, float32x4.splat(basis[1].x)),
                                         SIMD.mul(z, float32x4.splat(basis[2].x))));
            var diry = SIMD.add(SIMD.mul(x, float32x4.splat(basis[0].y)),
                                SIMD.add(SIMD.mul(y, float32x4.splat(basis[1].y)),
                                         SIMD.mul(z, float32x4.splat(basis[2].y))));
            var dirz = SIMD.add(SIMD.mul(x, float32x4.splat(basis[0].z)),
                                SIMD.add(SIMD.mul(y, float32x4.splat(basis[1].z)),
                                         SIMD.mul(z, float32x4.splat(basis[2].z))));
            
            var orgx = float32x4.splat(p.x);
            var orgy = float32x4.splat(p.y);
            var orgz = float32x4.splat(p.z);
            
            var occIsect = {
                t: float32x4.splat(1e17),
                hit: uint32x4.splat(0),
                p: {
                    x: float32x4.zero(),
                    y: float32x4.zero(),
                    z: float32x4.zero()
                },
                n: {
                    x: float32x4.zero(),
                    y: float32x4.zero(),
                    z: float32x4.zero()
                }
            };
            
            ray_sphere_intersect_simd(occIsect, dirx, diry, dirz, orgx, orgy, orgz, spheres[0]);
            ray_sphere_intersect_simd(occIsect, dirx, diry, dirz, orgx, orgy, orgz, spheres[1]);
            ray_sphere_intersect_simd(occIsect, dirx, diry, dirz, orgx, orgy, orgz, spheres[2]);
            ray_plane_intersect_simd (occIsect, dirx, diry, dirz, orgx, orgy, orgz, plane);
            
            occlusionx4 = SIMD.add(occlusionx4, 
                                   SIMD.bitsToFloat32x4(SIMD.and(occIsect.hit, SIMD.bitsToUint32x4(float32x4.splat(1)))));
            
        }
    }
    
    occlusion = occlusionx4.x + occlusionx4.y + occlusionx4.z + occlusionx4.w;
    
    occlusion = (ntheta * nphi - occlusion) / (ntheta * nphi);
    
    col.x = occlusion;
    col.y = occlusion;
    col.z = occlusion;
}

function render(img, w, h, nsubsamples) 
{
    
    var fimg = new Float32Array(w * h * 3);
    
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            
            for (var v = 0; v < nsubsamples; v++) {
                for (var u = 0; u < nsubsamples; u++) {
                    
                    var px = (x + (u/nsubsamples) - (w/2)) / (w/2);
                    var py = -(y + (v/nsubsamples) - (h/2)) / (h/2);
                    
                    var ray = {
                        org: { x:0, y:0, z:0 },
                        dir: { x:px, y:py, z:-1 }
                    };
                    vnormalize(ray.dir);
                    
                    var isect = {
                        t: 1e17,
                        hit: 0,
                        p: { x:0, y:0, z:0 },
                        n: { x:0, y:0, z:0 }
                    };
                    
                    ray_sphere_intersect(isect, ray, spheres[0]);
                    ray_sphere_intersect(isect, ray, spheres[1]);
                    ray_sphere_intersect(isect, ray, spheres[2]);
                    ray_plane_intersect(isect, ray, plane);
                    
                    if (isect.hit) {
                        var col = {};
                        ambient_occlusion(col, isect);
                        
                        fimg[3 * (y * w + x) + 0] += col.x;
                        fimg[3 * (y * w + x) + 1] += col.y;
                        fimg[3 * (y * w + x) + 2] += col.z;
                    }
                    
                }
            }
            
            fimg[3 * (y * w + x) + 0] /= (nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 1] /= (nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 2] /= (nsubsamples * nsubsamples);
            
            img[4 * (y * w + x) + 0] = fimg[3 *(y * w + x) + 0] * 255.5;
            img[4 * (y * w + x) + 1] = fimg[3 *(y * w + x) + 1] * 255.5;
            img[4 * (y * w + x) + 2] = fimg[3 *(y * w + x) + 2] * 255.5;
            img[4 * (y * w + x) + 3] = 255;
                        
        }
    }
}

function init_scene() {
    spheres = new Array();
    spheres[0] = {
        center: {
            x: -2.0,
            y: 0.0,
            z: -3.5
        },
        radius: 0.5
    };
    spheres[1] = {
        center: {
            x: -0.5,
            y: 0.0,
            z: -3.0
        },
        radius: 0.5
    };    
    spheres[2] = {
        center: {
            x: 1.0,
            y: 0.0,
            z: -2.2
        },
        radius: 0.5
    };
    plane = {
        p: {
            x: 0.0,
            y: -0.5,
            z: 0.0
        },
        n: {
            x: 0.0,
            y: 1.0,
            z: 0.0
        }
    };
}

function main() {    
    var ctx = document.getElementById('canvas').getContext('2d');
    ctx.canvas.width = WIDTH;
    ctx.canvas.height = HEIGHT;
    var imgData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
    var img = imgData.data;
    init_scene();
    console.log("rendering...");
    var timer = performance.now();
    render(img, WIDTH, HEIGHT, NSUBSAMPLES);
    timer = performance.now() - timer;
    console.log('...done rendering (' + timer.toFixed(2) + ' ms)');
    ctx.putImageData(imgData, 0, 0);
}