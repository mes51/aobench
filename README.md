Original sequential C version of aobench from https://code.google.com/p/aobench/

Live hosting of JS versions of aobench: [sequential] (http://wahbahdoo.github.io/aobench/js/ao.html); [SIMD] (http://wahbahdoo.github.io/aobench/js/ao_simd.html) (polyfill, i.e. slow!)

Multiple versions of SIMD implementations in C:
* \_aos: array of structures; rgb values mapped to 3 out of 4 lanes of vector ops
* \_soa: structure of arrays; hotspot function ambient\_occlusion is SIMDized to operate on 4 iterations of the loop at once
* \_soa\_nossemath: soa, but serial sin/cos rather than using a [SSE math library] (http://gruntthepeon.free.fr/ssemath/sse_mathfun.h)
* \_soa\_inlined: soa, but some more inlining and code refactoring for non-simd-related optimization

JS SIMD implementation is equivalent to \_soa\_nossemath.

