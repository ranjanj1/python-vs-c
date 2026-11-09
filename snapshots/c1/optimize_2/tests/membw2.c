/* Second-round kernel microbenchmark (profiling evidence). Packed R-row panels with
 * A accumulator sets per row, and a "column-split" variant for long rows (w2 shape):
 * the j loop is split into K-blocks of 512 cols processed as separate panels.
 * build: cc -O3 -mcpu=apple-m1 -o /tmp/membw2 tests/membw2.c ; run: /tmp/membw2 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
static double now(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * (double)tb.numer / tb.denom * 1e-9; }
static float *x, *out;
static void pack(float *dst, const float *src, int n, int d, int R) {
    for (int i = 0; i < d; i += R) for (int j = 0; j < n; j += 16) for (int r = 0; r < R; r++)
        { memcpy(dst, src + (size_t)(i + r) * n + j, 64); dst += 16; }
}
/* R rows, 2 accumulator sets per row (alternate 16-col blocks): 8*R accumulators */
#define MM2(R) \
static void mm2_p##R(float *o, const float *w, int n, int d) { \
    for (int i = 0; i < d; i += R) { \
        float32x4_t a[R][8]; \
        for (int r = 0; r < R; r++) for (int q = 0; q < 8; q++) a[r][q] = vdupq_n_f32(0); \
        for (int j = 0; j < n; j += 32) { \
            float32x4x4_t x0 = vld1q_f32_x4(x + j); \
            for (int r = 0; r < R; r++) { float32x4x4_t ww = vld1q_f32_x4(w); w += 16; \
                a[r][0] = vfmaq_f32(a[r][0], ww.val[0], x0.val[0]); a[r][1] = vfmaq_f32(a[r][1], ww.val[1], x0.val[1]); \
                a[r][2] = vfmaq_f32(a[r][2], ww.val[2], x0.val[2]); a[r][3] = vfmaq_f32(a[r][3], ww.val[3], x0.val[3]); } \
            float32x4x4_t x1 = vld1q_f32_x4(x + j + 16); \
            for (int r = 0; r < R; r++) { float32x4x4_t ww = vld1q_f32_x4(w); w += 16; \
                a[r][4] = vfmaq_f32(a[r][4], ww.val[0], x1.val[0]); a[r][5] = vfmaq_f32(a[r][5], ww.val[1], x1.val[1]); \
                a[r][6] = vfmaq_f32(a[r][6], ww.val[2], x1.val[2]); a[r][7] = vfmaq_f32(a[r][7], ww.val[3], x1.val[3]); } \
        } \
        for (int r = 0; r < R; r++) { float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][4]), vaddq_f32(a[r][1], a[r][5])), \
                                                     vaddq_f32(vaddq_f32(a[r][2], a[r][6]), vaddq_f32(a[r][3], a[r][7]))); \
            o[i + r] = vaddvq_f32(s); } \
    } }
MM2(1) MM2(2) MM2(4)
/* 4-row packed, 4 acc per row (16 acc) = previous best */
static void mm_p4(float *o, const float *w, int n, int d) {
    for (int i = 0; i < d; i += 4) {
        float32x4_t a[4][4];
        for (int r = 0; r < 4; r++) a[r][0] = a[r][1] = a[r][2] = a[r][3] = vdupq_n_f32(0);
        for (int j = 0; j < n; j += 16) {
            float32x4x4_t xx = vld1q_f32_x4(x + j);
            for (int r = 0; r < 4; r++) { float32x4x4_t ww = vld1q_f32_x4(w); w += 16;
                a[r][0] = vfmaq_f32(a[r][0], ww.val[0], xx.val[0]); a[r][1] = vfmaq_f32(a[r][1], ww.val[1], xx.val[1]);
                a[r][2] = vfmaq_f32(a[r][2], ww.val[2], xx.val[2]); a[r][3] = vfmaq_f32(a[r][3], ww.val[3], xx.val[3]); }
        }
        for (int r = 0; r < 4; r++) o[i + r] = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3])));
    }
}
typedef void (*mmfn)(float *, const float *, int, int);
int main(void) {
    int shapes[3][2] = { {2048, 768}, {768, 2048}, {768, 768} }; /* w2 first to control for order */
    int mult[3] = {12, 12, 48};
    const char *sn[3] = {"w2-like 768x2048 x12", "w1-like 2048x768 x12", "768x768 x48"};
    x = aligned_alloc(128, 8192 * 4); out = aligned_alloc(128, 8192 * 4);
    for (int i = 0; i < 2048; i++) x[i] = 1.0f / (i + 1);
    struct { const char *name; mmfn f; int R; } v[] = { {"packed R=4 acc4", mm_p4, 4}, {"packed R=1 acc8", mm2_p1, 1},
        {"packed R=2 acc8", mm2_p2, 2}, {"packed R=4 acc8", mm2_p4, 4} };
    volatile float sink = 0;
    for (int round = 0; round < 2; round++)
    for (int s = 0; s < 3; s++) {
        int n = shapes[s][0], d = shapes[s][1]; size_t per = (size_t)n * d, cnt = per * mult[s];
        float *src = aligned_alloc(128, cnt * 4), *pk = aligned_alloc(128, cnt * 4);
        for (size_t i = 0; i < cnt; i++) src[i] = (float)(i % 97) * 0.01f;
        printf("%s (%.0f MB) round %d\n", sn[s], cnt * 4 / 1e6, round);
        for (int k = 0; k < 4; k++) {
            for (int m = 0; m < mult[s]; m++) pack(pk + m * per, src + m * per, n, d, v[k].R);
            double best = 1e9;
            for (int rep = 0; rep < 7; rep++) { double t = now(); for (int m = 0; m < mult[s]; m++) v[k].f(out, pk + m * per, n, d); t = now() - t; if (t < best) best = t; sink += out[0]; }
            printf("  %-18s %6.1f GB/s\n", v[k].name, cnt * 4 / best / 1e9);
        }
        free(src); free(pk);
    }
    return sink == 12345;
}
