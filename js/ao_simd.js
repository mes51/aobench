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


function ray_sphere_intersect(isectA, isectB, ray, sphere) 
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
    if ((t > 0) && (t < isectA.t)) {
      
      isectA.t = t;
      isectA.hit = 1;
      
      isectB.p.x = ray.org.x + ray.dir.x * t;
      isectB.p.y = ray.org.y + ray.dir.y * t;
      isectB.p.z = ray.org.z + ray.dir.z * t;
      
      isectB.n.x = isectB.p.x - sphere.center.x;
      isectB.n.y = isectB.p.y - sphere.center.y;
      isectB.n.z = isectB.p.z - sphere.center.z;
      
      vnormalize(isectB.n);
    }
  }
  
}


function ray_sphere_intersect_simd(isectA, isectB, dirx, diry, dirz, orgx, orgy, orgz, sphere) 
{
  var rsx = SIMD.float32x4.sub(orgx, float32x4.splat(sphere.center.x));
  var rsy = SIMD.float32x4.sub(orgy, float32x4.splat(sphere.center.y));
  var rsz = SIMD.float32x4.sub(orgz, float32x4.splat(sphere.center.z));
  
  var B = SIMD.float32x4.add(SIMD.float32x4.mul(rsx, dirx),
                   SIMD.float32x4.add(SIMD.float32x4.mul(rsy, diry), SIMD.float32x4.mul(rsz, dirz)));
  var C = SIMD.float32x4.sub(SIMD.float32x4.add(SIMD.float32x4.mul(rsx, rsx),
                            SIMD.float32x4.add(SIMD.float32x4.mul(rsy, rsy), SIMD.float32x4.mul(rsz, rsz))),
                   float32x4.splat(sphere.radius * sphere.radius));
  var D = SIMD.float32x4.sub(SIMD.float32x4.mul(B, B), C);
  
  var cond1 = SIMD.float32x4.greaterThan(D, float32x4.zero());
  if (cond1.signMask) {
    var t2 = SIMD.int32x4.bitsToFloat32x4(SIMD.int32x4.and(cond1, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.sub(SIMD.float32x4.neg(B), SIMD.float32x4.sqrt(D)))));
    var cond2 = SIMD.int32x4.and(SIMD.float32x4.greaterThan(t2, float32x4.zero()),
                                 SIMD.float32x4.lessThan(t2, isectA.t));
    if (cond2.signMask) {
      isectA.t = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(
                     SIMD.int32x4.and(
                       cond2,
                       SIMD.float32x4.bitsToInt32x4(t2)),
                     SIMD.int32x4.and(
                       SIMD.int32x4.not(cond2),
                       SIMD.float32x4.bitsToInt32x4(isectA.t))));
      isectA.hit = SIMD.int32x4.or(cond2, isectA.hit);
      isectB.p.x = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(dirx, isectA.t)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.x))));
      isectB.p.y = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(diry, isectA.t)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.y))));
      isectB.p.z = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(dirz, isectA.t)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.z))));
      
      isectB.n.x = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.sub(isectB.p.x, float32x4.splat(sphere.center.x)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.x))));
      isectB.n.y = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.sub(isectB.p.y, float32x4.splat(sphere.center.y)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.y))));
      isectB.n.z = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.sub(isectB.p.z, float32x4.splat(sphere.center.z)))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.z))));
      
      var lengths = SIMD.float32x4.sqrt(SIMD.float32x4.add(SIMD.float32x4.mul(isectB.n.x, isectB.n.x),
                                        SIMD.float32x4.add(SIMD.float32x4.mul(isectB.n.y, isectB.n.y),
                                                           SIMD.float32x4.mul(isectB.n.z, isectB.n.z))));
      var cond3 = SIMD.float32x4.greaterThan(SIMD.float32x4.abs(lengths), float32x4.splat(1e-17));
      isectB.n.x = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond3, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.div(isectB.n.x, lengths))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond3), SIMD.float32x4.bitsToInt32x4(isectB.n.x))));
      isectB.n.y = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond3, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.div(isectB.n.y, lengths))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond3), SIMD.float32x4.bitsToInt32x4(isectB.n.y))));
      isectB.n.z = SIMD.int32x4.bitsToFloat32x4(
                     SIMD.int32x4.or(SIMD.int32x4.and(cond3, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.div(isectB.n.z, lengths))),
                     SIMD.int32x4.and(SIMD.int32x4.not(cond3), SIMD.float32x4.bitsToInt32x4(isectB.n.z))));
    }
  }
}


function ray_plane_intersect(isectA, isectB, ray, plane) 
{
  var d = -vdot(plane.p, plane.n);
  var v = vdot(ray.dir, plane.n);
  
  if (Math.abs(v) < 1e-17) return;
  
  var t = -(vdot(ray.org, plane.n) + d) / v;
  
  if ((t > 0) && (t < isectA.t)) {
    isectA.t = t;
    isectA.hit = 1;
    isectB.p.x = ray.org.x + ray.dir.x * t;
    isectB.p.y = ray.org.y + ray.dir.y * t;
    isectB.p.z = ray.org.z + ray.dir.z * t;
    isectB.n = plane.n;
  }
}


function ray_plane_intersect_simd(isectA, isectB, dirx, diry, dirz, orgx, orgy, orgz, plane) 
{
  var d = SIMD.float32x4.neg(SIMD.float32x4.add(SIMD.float32x4.mul(float32x4.splat(plane.p.x), float32x4.splat(plane.n.x)),
                             SIMD.float32x4.add(SIMD.float32x4.mul(float32x4.splat(plane.p.y), float32x4.splat(plane.n.y)),
                                      SIMD.float32x4.mul(float32x4.splat(plane.p.z), float32x4.splat(plane.n.z)))));
  var v = SIMD.float32x4.add(SIMD.float32x4.mul(dirx, float32x4.splat(plane.n.x)),
                   SIMD.float32x4.add(SIMD.float32x4.mul(diry, float32x4.splat(plane.n.y)),
                            SIMD.float32x4.mul(dirz, float32x4.splat(plane.n.z))));
  
  var cond1 = SIMD.float32x4.greaterThan(SIMD.float32x4.abs(v), float32x4.splat(1e-17));
  var dp = SIMD.float32x4.add(SIMD.float32x4.mul(orgx, float32x4.splat(plane.n.x)),
                    SIMD.float32x4.add(SIMD.float32x4.mul(orgy, float32x4.splat(plane.n.y)),
                             SIMD.float32x4.mul(orgz, float32x4.splat(plane.n.z))));
  var t2 = SIMD.int32x4.bitsToFloat32x4(SIMD.int32x4.and(cond1, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.div(SIMD.float32x4.neg(SIMD.float32x4.add(dp, d)), v))));
  var cond2 = SIMD.int32x4.and(SIMD.float32x4.greaterThan(t2, float32x4.zero()), SIMD.float32x4.lessThan(t2, isectA.t));
  if (cond2.signMask) {
    isectA.t = SIMD.int32x4.bitsToFloat32x4(SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(t2)),
                                           SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectA.t))));
    isectA.hit = SIMD.int32x4.or(cond2, isectA.hit);
    isectB.p.x = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(dirx, isectA.t)))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.x))));
    isectB.p.y = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(diry, isectA.t)))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.y))));
    isectB.p.z = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(SIMD.float32x4.add(orgx, SIMD.float32x4.mul(dirz, isectA.t)))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.p.z))));
    
    isectB.n.x = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(float32x4.splat(plane.n.x))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.x))));
    isectB.n.y = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(float32x4.splat(plane.n.y))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.y))));
    isectB.n.z = SIMD.int32x4.bitsToFloat32x4(
                   SIMD.int32x4.or(SIMD.int32x4.and(cond2, SIMD.float32x4.bitsToInt32x4(float32x4.splat(plane.n.z))),
                   SIMD.int32x4.and(SIMD.int32x4.not(cond2), SIMD.float32x4.bitsToInt32x4(isectB.n.z))));
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
    for (i = 0; i < nphi; i += 4) {
      var theta = SIMD.float32x4.sqrt(float32x4(Math.random(), Math.random(), Math.random(), Math.random()));
      var phi0 = 2 * Math.PI * Math.random();
      var phi1 = 2 * Math.PI * Math.random();
      var phi2 = 2 * Math.PI * Math.random();
      var phi3 = 2 * Math.PI * Math.random();
      var sinphi = float32x4(Math.sin(phi0), Math.sin(phi1), Math.sin(phi2), Math.sin(phi3));
      var cosphi = float32x4(Math.cos(phi0), Math.cos(phi1), Math.cos(phi2), Math.cos(phi3));
      
      var x = SIMD.float32x4.mul(cosphi, theta);
      var y = SIMD.float32x4.mul(sinphi, theta);
      var z = SIMD.float32x4.sqrt(SIMD.float32x4.sub(float32x4.splat(1), SIMD.float32x4.mul(theta, theta)));
      
      var dirx = SIMD.float32x4.add(SIMD.float32x4.mul(x, float32x4.splat(basis[0].x)),
                          SIMD.float32x4.add(SIMD.float32x4.mul(y, float32x4.splat(basis[1].x)),
                                   SIMD.float32x4.mul(z, float32x4.splat(basis[2].x))));
      var diry = SIMD.float32x4.add(SIMD.float32x4.mul(x, float32x4.splat(basis[0].y)),
                          SIMD.float32x4.add(SIMD.float32x4.mul(y, float32x4.splat(basis[1].y)),
                                   SIMD.float32x4.mul(z, float32x4.splat(basis[2].y))));
      var dirz = SIMD.float32x4.add(SIMD.float32x4.mul(x, float32x4.splat(basis[0].z)),
                          SIMD.float32x4.add(SIMD.float32x4.mul(y, float32x4.splat(basis[1].z)),
                                   SIMD.float32x4.mul(z, float32x4.splat(basis[2].z))));
      
      var orgx = float32x4.splat(p.x);
      var orgy = float32x4.splat(p.y);
      var orgz = float32x4.splat(p.z);
      
      var occIsectA = {
        t: float32x4.splat(1e17),
        hit: int32x4.splat(0)
      };
      var occIsectB = {
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
      
      ray_sphere_intersect_simd(occIsectA, occIsectB, dirx, diry, dirz, orgx, orgy, orgz, spheres[0]);
      ray_sphere_intersect_simd(occIsectA, occIsectB, dirx, diry, dirz, orgx, orgy, orgz, spheres[1]);
      ray_sphere_intersect_simd(occIsectA, occIsectB, dirx, diry, dirz, orgx, orgy, orgz, spheres[2]);
      ray_plane_intersect_simd (occIsectA, occIsectB, dirx, diry, dirz, orgx, orgy, orgz, plane);
      
      occlusionx4 = SIMD.float32x4.add(
                      occlusionx4,
                      SIMD.int32x4.bitsToFloat32x4(
                        SIMD.int32x4.and(
                          occIsectA.hit, SIMD.float32x4.bitsToInt32x4(float32x4.splat(1)))));
      
    }
  }
  
  occlusion = occlusionx4.x + occlusionx4.y + occlusionx4.z + occlusionx4.w;
  
  occlusion = (ntheta * nphi - occlusion) / (ntheta * nphi);
  
  col.x = occlusion;
  col.y = occlusion;
  col.z = occlusion;

  return col;
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
          
          var isectA = {
            t: 1e17,
            hit: 0
          };
          var isectB = {
            p: { x:0, y:0, z:0 },
            n: { x:0, y:0, z:0 }
          };
                    
          ray_sphere_intersect(isectA, isectB, ray, spheres[0]);
          ray_sphere_intersect(isectA, isectB, ray, spheres[1]);
          ray_sphere_intersect(isectA, isectB, ray, spheres[2]);
          ray_plane_intersect(isectA, isectB, ray, plane);
                    
          if (isectA.hit) {
            var col = {};
            ambient_occlusion(col, isectB);
            
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


function init_scene() 
{
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