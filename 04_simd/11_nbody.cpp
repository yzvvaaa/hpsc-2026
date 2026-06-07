#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 x_i_vec = _mm512_set1_ps(x[i]);
    __m512 y_i_vec = _mm512_set1_ps(y[i]);
    
    __m512 rx = _mm512_sub_ps(x_i_vec, xvec);
    __m512 ry = _mm512_sub_ps(y_i_vec, yvec);
    
    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    __m512 r = _mm512_sqrt_ps(r2);
    
    __mmask16 mask = ~(1 << i);
    __m512 r_safe = _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0f), r);
    
    __m512 r3 = _mm512_mul_ps(_mm512_mul_ps(r_safe, r_safe), r_safe);
    __m512 coeff = _mm512_div_ps(mvec, r3);
    coeff = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), coeff);
    
    __m512 fx_contrib = _mm512_mul_ps(rx, coeff);
    __m512 fy_contrib = _mm512_mul_ps(ry, coeff);
    
    fx[i] = -_mm512_reduce_add_ps(fx_contrib);
    fy[i] = -_mm512_reduce_add_ps(fy_contrib);
    
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
