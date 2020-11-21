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
  
  for (var j = 0; j < ntheta; j++) {
    for (var i = 0; i < nphi; i++) {
      var theta = Math.sqrt(Math.random());
      var phi = 2 * Math.PI * Math.random();
      
      var x = Math.cos(phi) * theta;
      var y = Math.sin(phi) * theta;
      var z = Math.sqrt(1 - theta * theta);
      
      var rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      var ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      var rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;
      
      var ray = {
        org: p,
        dir: {
          x: rx,
          y: ry,
          z: rz
        }
      };
      
      var occIsectA = {
        t: 1e17,
        hit: 0
      }
      var occIsectB = {
        p: { x:0, y:0, z:0 },
        n: { x:0, y:0, z:0 }
      };
      
      ray_sphere_intersect(occIsectA, occIsectB, ray, spheres[0]);
      ray_sphere_intersect(occIsectA, occIsectB, ray, spheres[1]);
      ray_sphere_intersect(occIsectA, occIsectB, ray, spheres[2]);
      ray_plane_intersect(occIsectA, occIsectB, ray, plane);
      
      if (occIsectA.hit) occlusion += 1.0;
      
    }
  }
  
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

function main() 
{
  var ctx = document.getElementById('canvas').getContext('2d');
  ctx.canvas.width = WIDTH;
  ctx.canvas.height = HEIGHT;
  var imgData = ctx.getImageData(0, 0, WIDTH, HEIGHT);
  var img = imgData.data;
  init_scene();    console.log("rendering...");
  var timer = performance.now();
  render(img, WIDTH, HEIGHT, NSUBSAMPLES);
  timer = performance.now() - timer;
  document.getElementById('time').textContent = timer.toFixed(2);
  console.log('...done rendering (' + timer.toFixed(2) + ' ms)');
  ctx.putImageData(imgData, 0, 0);
}