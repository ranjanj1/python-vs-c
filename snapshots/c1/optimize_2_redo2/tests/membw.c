/* Memory-bandwidth microbenchmark used for profiling the matmul kernel.
 * Compares, over model-sized weight matrices (12 layers of 768x768 and 2048x768 / 768x2048):
 *   roofline : pure sequential read+sum of the whole buffer
 *   rowmajor : current kernel, 2 row pointers, 64B loads
 *   packedR  : rows pre-packed into R-row panels (16-float blocks interleaved) -> 1 sequential stream
 * build: cc -O3 -mcpu=apple-m1 -o /tmp/membw tests/membw.c ; run: /tmp/membw */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
static double now(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * (double)tb.numer / tb.denom * 1e-9; }
#define NL 12
static float *x, *out;
static float roof(const float *w, size_t n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (size_t j = 0; j < n; j += 32) {
        float32x4x4_t u = vld1q_f32_x4(w + j), v = vld1q_f32_x4(w + j + 16);
        a0 = vaddq_f32(a0, u.val[0]); a1 = vaddq_f32(a1, u.val[1]); a2 = vaddq_f32(a2, u.val[2]); a3 = vaddq_f32(a3, u.val[3]);
        a4 = vaddq_f32(a4, v.val[0]); a5 = vaddq_f32(a5, v.val[1]); a6 = vaddq_f32(a6, v.val[2]); a7 = vaddq_f32(a7, v.val[3]);
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)), vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7))));
}
static void mm_row(float *o, const float *w, int n, int d) {
    for (int i = 0; i < d; i += 2) {
        const float *r0 = w + (size_t)i * n, *r1 = r0 + n;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int j = 0; j < n; j += 16) {
            float32x4x4_t xx = vld1q_f32_x4(x + j), w0 = vld1q_f32_x4(r0 + j), w1 = vld1q_f32_x4(r1 + j);
            a0 = vfmaq_f32(a0, w0.val[0], xx.val[0]); a1 = vfmaq_f32(a1, w0.val[1], xx.val[1]);
            a2 = vfmaq_f32(a2, w0.val[2], xx.val[2]); a3 = vfmaq_f32(a3, w0.val[3], xx.val[3]);
            b0 = vfmaq_f32(b0, w1.val[0], xx.val[0]); b1 = vfmaq_f32(b1, w1.val[1], xx.val[1]);
            b2 = vfmaq_f32(b2, w1.val[2], xx.val[2]); b3 = vfmaq_f32(b3, w1.val[3], xx.val[3]);
        }
        o[i] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        o[i + 1] = vaddvq_f32(vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3)));
    }
}
/* packed: for each panel of R rows, for each 16-col block j: R consecutive 16-float blocks */
static void pack(float *dst, const float *src, int n, int d, int R) {
    for (int i = 0; i < d; i += R) for (int j = 0; j < n; j += 16) for (int r = 0; r < R; r++)
        { memcpy(dst, src + (size_t)(i + r) * n + j, 64); dst += 16; }
}
#define MM_PACKED(R) \
static void mm_p##R(float *o, const float *w, int n, int d) { \
    for (int i = 0; i < d; i += R) { \
        float32x4_t a[R][4]; \
        for (int r = 0; r < R; r++) a[r][0] = a[r][1] = a[r][2] = a[r][3] = vdupq_n_f32(0); \
        for (int j = 0; j < n; j += 16) { \
            float32x4x4_t xx = vld1q_f32_x4(x + j); \
            for (int r = 0; r < R; r++) { float32x4x4_t ww = vld1q_f32_x4(w); w += 16; \
                a[r][0] = vfmaq_f32(a[r][0], ww.val[0], xx.val[0]); a[r][1] = vfmaq_f32(a[r][1], ww.val[1], xx.val[1]); \
                a[r][2] = vfmaq_f32(a[r][2], ww.val[2], xx.val[2]); a[r][3] = vfmaq_f32(a[r][3], ww.val[3], xx.val[3]); } \
        } \
        for (int r = 0; r < R; r++) o[i + r] = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3]))); \
    } }
MM_PACKED(1) MM_PACKED(2) MM_PACKED(4) MM_PACKED(8)
typedef void (*mmfn)(float *, const float *, int, int);
int main(void) {
    int shapes[3][2] = { {768, 768}, {768, 2048}, {2048, 768} }; /* (n,d): wq-like, w1-like, w2-like */
    size_t per[3]; float *buf[3], *pk[4][3];
    x = aligned_alloc(128, 8192 * 4); out = aligned_alloc(128, 8192 * 4);
    for (int i = 0; i < 2048; i++) x[i] = 1.0f / (i + 1);
    for (int s = 0; s < 3; s++) {
        per[s] = (size_t)shapes[s][0] * shapes[s][1];
        buf[s] = aligned_alloc(128, per[s] * NL * 4 * (s == 0 ? 4 : 1));
        for (size_t i = 0; i < per[s] * NL * (s == 0 ? 4 : 1); i++) buf[s][i] = (float)(i % 97) * 0.01f;
    }
    int Rs[4] = {1, 2, 4, 8}; mmfn pf[4] = {mm_p1, mm_p2, mm_p4, mm_p8};
    for (int k = 0; k < 4; k++) for (int s = 0; s < 3; s++) {
        size_t cnt = per[s] * NL * (s == 0 ? 4 : 1);
        pk[k][s] = aligned_alloc(128, cnt * 4);
        int nm = cnt / per[s];
        for (int m = 0; m < nm; m++) pack(pk[k][s] + m * per[s], buf[s] + m * per[s], shapes[s][0], shapes[s][1], Rs[k]);
    }
    const char *sn[3] = {"768x768 x48 (qkvo)", "w1-like 2048x768 x12", "w2-like 768x2048 x12"};
    volatile float sink = 0;
    for (int s = 0; s < 3; s++) {
        int nm = s == 0 ? 4 * NL : NL; double bytes = (double)per[s] * nm * 4, best;
        printf("%s (%.0f MB)\n", sn[s], bytes / 1e6);
        best = 1e9; for (int rep = 0; rep < 7; rep++) { double t = now(); sink += roof(buf[s], per[s] * nm); t = now() - t; if (t < best) best = t; }
        printf("  roofline read      %6.1f GB/s\n", bytes / best / 1e9);
        best = 1e9; for (int rep = 0; rep < 7; rep++) { double t = now(); for (int m = 0; m < nm; m++) mm_row(out, buf[s] + m * per[s], shapes[s][0], shapes[s][1]); t = now() - t; if (t < best) best = t; sink += out[0]; }
        printf("  rowmajor 2-row     %6.1f GB/s\n", bytes / best / 1e9);
        for (int k = 0; k < 4; k++) {
            best = 1e9; for (int rep = 0; rep < 7; rep++) { double t = now(); for (int m = 0; m < nm; m++) pf[k](out, pk[k][s] + m * per[s], shapes[s][0], shapes[s][1]); t = now() - t; if (t < best) best = t; sink += out[0]; }
            printf("  packed R=%d         %6.1f GB/s\n", Rs[k], bytes / best / 1e9);
        }
    }
    return sink == 12345;
}
