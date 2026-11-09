/* C1 port of the pinned llama2.c reference (reference/run.c).
 * Plain C + compiler flags + ARM NEON intrinsics. No math library: exp, sqrt,
 * sin/cos/pow (for RoPE tables) are implemented here or via CPU instructions.
 * Single thread. float32 weights and activations, batch size 1, one token per
 * forward pass, KV cache identical to the reference.
 * Protocol: see INTERFACE.md. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <arm_neon.h>

/* Optional section profiler: build with -DPROFILE (see build.sh). Compiled
 * out completely in the normal build. Summary goes to stderr at QUIT. */
#ifdef PROFILE
#include <mach/mach_time.h>
enum { P_RMS, P_QKV, P_ROPE, P_ATT, P_WO, P_W13, P_SILU, P_W2, P_RESID, P_CLS, P_ARGMAX,
       P_ENCODE, P_LOGIO, P_N };
static const char *prof_name[P_N] = { "rmsnorm+embed", "matmul qkv", "rope+kv store", "attention",
    "matmul wo", "matmul w1+w3", "swiglu", "matmul w2", "residual", "matmul classifier",
    "argmax", "encode", "logits file write" };
static uint64_t prof_t[P_N], prof_last, prof_forwards;
#define PSTART() (prof_last = mach_absolute_time())
#define PMARK(id) do { uint64_t _n = mach_absolute_time(); prof_t[id] += _n - prof_last; prof_last = _n; } while (0)
static void prof_report(void) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    double tot = 0; for (int i = 0; i < P_N; i++) tot += prof_t[i];
    fprintf(stderr, "PROFILE forwards=%llu total=%.1f ms\n", prof_forwards, tot * tb.numer / tb.denom / 1e6);
    for (int i = 0; i < P_N; i++) {
        double ms = prof_t[i] * (double)tb.numer / tb.denom / 1e6;
        fprintf(stderr, "  %-20s %9.1f ms %5.1f%%  %7.3f ms/forward\n", prof_name[i], ms,
                tot ? 100.0 * prof_t[i] / tot : 0, prof_forwards ? ms / prof_forwards : 0);
    }
}
/* load-phase timestamps */
static uint64_t lt_last;
static void load_mark(const char *what) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    uint64_t n = mach_absolute_time();
    if (lt_last) fprintf(stderr, "LOAD %-28s %7.1f ms\n", what, (n - lt_last) * (double)tb.numer / tb.denom / 1e6);
    lt_last = n;
}
#define LMARK(what) load_mark(what)
#else
#define LMARK(what) ((void)0)
#define PSTART() ((void)0)
#define PMARK(id) ((void)0)
#endif

/* ------------------------------------------------------------------------- */
/* math helpers (no libm)                                                     */

static inline float my_sqrtf(float x) { return __builtin_sqrtf(x); } /* fsqrt instruction */

/* float32 exp: Cody-Waite reduction x = n*ln2 + r, |r| <= ln2/2, degree-6
 * polynomial, scale by 2^n with split to cover subnormal/overflow range. */
static inline float32x4_t vexpq(float32x4_t x) {
    const float32x4_t hi = vdupq_n_f32(88.72283935546875f);
    const float32x4_t lo = vdupq_n_f32(-103.97208f);
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t ln2hi = vdupq_n_f32(0.693145751953125f);
    const float32x4_t ln2lo = vdupq_n_f32(1.428606765330187045e-06f);
    uint32x4_t overflow = vcgtq_f32(x, hi);
    uint32x4_t underflow = vcltq_f32(x, lo);
    float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t r = vfmsq_f32(x, n, ln2hi);
    r = vfmsq_f32(r, n, ln2lo);
    /* exp(r) - 1 - r ~= r^2 * P(r) */
    float32x4_t p = vdupq_n_f32(1.0f / 5040.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t e = vaddq_f32(vfmaq_f32(r, p, r2), vdupq_n_f32(1.0f));
    /* scale by 2^n as 2^n1 * 2^n2 so that n in [-150, 128] works */
    int32x4_t ni = vcvtq_s32_f32(n);
    int32x4_t n1 = vshrq_n_s32(ni, 1);
    int32x4_t n2 = vsubq_s32(ni, n1);
    float32x4_t s1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n1, vdupq_n_s32(127)), 23));
    float32x4_t s2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n2, vdupq_n_s32(127)), 23));
    e = vmulq_f32(vmulq_f32(e, s1), s2);
    e = vbslq_f32(overflow, vdupq_n_f32(__builtin_inff()), e);
    e = vbslq_f32(underflow, vdupq_n_f32(0.0f), e);
    /* NaN in -> NaN out (comparisons above are false for NaN, rndn keeps NaN,
     * but cvt turns NaN into 0; restore it explicitly) */
    uint32x4_t isnan = vmvnq_u32(vceqq_f32(x, x));
    e = vbslq_f32(isnan, x, e);
    return e;
}

static inline float my_expf(float x) {
    return vgetq_lane_f32(vexpq(vdupq_n_f32(x)), 0);
}

/* double-precision helpers used only to build the RoPE tables at load time */
static double d_exp(double x) {
    /* x = k*ln2 + r, exp(r) by Taylor to 1e-17 */
    const double LN2 = 0.6931471805599453094;
    double kd = x / LN2;
    long k = (long)(kd < 0 ? kd - 0.5 : kd + 0.5);
    double r = x - (double)k * LN2;
    double term = 1.0, sum = 1.0;
    for (int i = 1; i < 30; i++) { term *= r / i; sum += term; }
    while (k > 0) { sum *= 2.0; k--; }
    while (k < 0) { sum *= 0.5; k++; }
    return sum;
}
static void d_sincos(double x, double *s, double *c) {
    /* reduce by pi/2 (args here are < ~1100, double reduction is ample) */
    const double PIO2_1 = 1.57079632673412561417e+00;
    const double PIO2_2 = 6.07710050650619224932e-11;
    const double PIO2_3 = 2.02226624871116645580e-21;
    double kd = x / 1.57079632679489661923;
    long k = (long)(kd < 0 ? kd - 0.5 : kd + 0.5);
    double r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
    double r2 = r * r, ts = r, tc = 1.0, ss = r, cc = 1.0;
    for (int i = 1; i < 15; i++) {
        ts *= -r2 / ((2 * i) * (2 * i + 1)); ss += ts;
        tc *= -r2 / ((2 * i - 1) * (2 * i)); cc += tc;
    }
    switch (((k % 4) + 4) % 4) {
        case 0: *s = ss;  *c = cc;  break;
        case 1: *s = cc;  *c = -ss; break;
        case 2: *s = -ss; *c = -cc; break;
        default: *s = -cc; *c = ss; break;
    }
}

/* ------------------------------------------------------------------------- */
/* model                                                                      */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

/* A matmul weight W (d rows, n cols). If packed, rows are stored in panels
 * of PR rows: for each panel, for each 16-column block, the PR row-segments
 * of 16 floats follow each other. All panels of all matrices are laid out
 * back-to-back in the exact order forward() consumes them, so one forward
 * pass reads the weights as a single sequential stream. If the shape does not
 * allow packing (n % 16 or d % PR), the matrix is stored row-major. */
#ifndef PR
#define PR 4 /* panel height; 2 and 4 measured equal, 8 slower, see NOTES.md */
#endif
typedef struct { const float *p; int n, d, packed; } Mat;

typedef struct {
    float *token_embedding_table;              /* only kept if not shared with wcls */
    float *rms_att_weight, *rms_ffn_weight, *rms_final_weight;
    Mat *wq, *wk, *wv, *wo, *w1, *w2, *w3;     /* per layer */
    Mat wcls;
    int shared;
} Weights;

static Config cfg;
static Weights W;
static float *packed_buf;
static int head_size, kv_dim, kv_mul;
static float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *key_cache, *value_cache;
static float *kcur, *vcur; /* this position's k and v before they are stored in the cache */
/* KV cache layout is head-major: [layer][kv_head][pos][head_size], so one head's
 * keys/values for positions 0..pos are contiguous. Same values, same positions,
 * written once per position, as in the reference; only the index order differs. */
static inline size_t kv_index(size_t l, int kh, int t) {
    return ((l * cfg.n_kv_heads + kh) * (size_t)cfg.seq_len + t) * head_size;
}
static float *rope_cos, *rope_sin; /* (seq_len, head_size/2) */

static void *xalloc_nozero(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 128, n ? n : 16) != 0) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void *xalloc(size_t n) {
    void *p = xalloc_nozero(n);
    memset(p, 0, n);
    return p;
}

static size_t mat_floats(int n, int d) { return (size_t)n * d; }

/* read nfl floats starting at float offset off (from the start of the file) */
static int model_fd;
static void read_floats(float *dst, size_t off, size_t nfl) {
    char *p = (char *)dst;
    size_t left = nfl * sizeof(float);
    off_t pos = (off_t)(off * sizeof(float));
    while (left) {
        size_t chunk = left > (1u << 30) ? (1u << 30) : left;
        ssize_t r = pread(model_fd, p, chunk, pos);
        if (r <= 0) { fprintf(stderr, "checkpoint read failed (truncated file?)\n"); exit(1); }
        p += r; pos += r; left -= (size_t)r;
    }
}

/* Load the d x n row-major matrix at float offset off into dst, in the packed
 * panel layout. Rows are read in L2-sized chunks (a multiple of PR rows) into
 * a staging buffer and packed from there; packing is row-local, so chunking
 * does not change the result. */
static float *stage;
static size_t stage_floats;
static Mat load_mat(float *dst, size_t off, int n, int d) {
    Mat m = { dst, n, d, (n % 16 == 0 && d % PR == 0) };
    if (!m.packed) { read_floats(dst, off, mat_floats(n, d)); return m; }
    int rows = (int)(stage_floats / (size_t)n) / PR * PR;
    if (rows < PR) rows = PR;
    for (int i0 = 0; i0 < d; i0 += rows) {
        int nr = d - i0 < rows ? d - i0 : rows;
        read_floats(stage, off + (size_t)i0 * n, (size_t)nr * n);
        for (int i = 0; i < nr; i += PR)
            for (int j = 0; j < n; j += 16)
                for (int r = 0; r < PR; r++) {
                    memcpy(dst, stage + (size_t)(i + r) * n + j, 16 * sizeof(float));
                    dst += 16;
                }
    }
    return m;
}

static float *load_aligned(size_t off, size_t nfl) {
    float *d = xalloc(nfl * sizeof(float));
    read_floats(d, off, nfl);
    return d;
}

static void load_model(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    model_fd = fd;
    off_t size = lseek(fd, 0, SEEK_END);
    if (size < (off_t)sizeof(Config)) { fprintf(stderr, "bad checkpoint\n"); exit(1); }
    LMARK("start");
    if (pread(fd, &cfg, sizeof(Config), 0) != (ssize_t)sizeof(Config)) { fprintf(stderr, "bad checkpoint\n"); exit(1); }
    W.shared = cfg.vocab_size > 0;
    if (cfg.vocab_size < 0) cfg.vocab_size = -cfg.vocab_size;
    head_size = cfg.dim / cfg.n_heads;
    kv_dim = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
    kv_mul = cfg.n_heads / cfg.n_kv_heads;

    /* checkpoint layout, exactly as reference memory_map_weights(); offsets in floats */
    unsigned long long L = cfg.n_layers, d = cfg.dim, h = cfg.hidden_dim;
    size_t p = sizeof(Config) / sizeof(float);
    size_t emb = p; p += (size_t)cfg.vocab_size * d;
    size_t rms_att = p; p += L * d;
    size_t wq = p; p += L * d * (cfg.n_heads * head_size);
    size_t wk = p; p += L * d * (cfg.n_kv_heads * head_size);
    size_t wv = p; p += L * d * (cfg.n_kv_heads * head_size);
    size_t wo = p; p += L * (cfg.n_heads * head_size) * d;
    size_t rms_ffn = p; p += L * d;
    size_t w1 = p; p += L * d * h;
    size_t w2 = p; p += L * h * d;
    size_t w3 = p; p += L * d * h;
    size_t rms_fin = p; p += d;
    p += (size_t)cfg.seq_len * head_size / 2;
    p += (size_t)cfg.seq_len * head_size / 2;
    size_t wcls = W.shared ? emb : p;
    if ((wcls + (size_t)cfg.vocab_size * d) * sizeof(float) > (size_t)size) {
        fprintf(stderr, "checkpoint too small\n"); exit(1);
    }

    W.rms_att_weight = load_aligned(rms_att, L * d);
    W.rms_ffn_weight = load_aligned(rms_ffn, L * d);
    W.rms_final_weight = load_aligned(rms_fin, d);
    if (!W.shared) W.token_embedding_table = load_aligned(emb, (size_t)cfg.vocab_size * d);

    int qd = cfg.n_heads * head_size;
    size_t per_layer = mat_floats(d, qd) + 2 * mat_floats(d, kv_dim) + mat_floats(qd, d)
                     + 2 * mat_floats(d, h) + mat_floats(h, d);
    size_t total = per_layer * L + mat_floats(d, cfg.vocab_size);
    LMARK("header+small reads");
    /* every float of packed_buf is written by load_mat, so no zeroing */
    packed_buf = xalloc_nozero(total * sizeof(float));
#ifndef STAGE_MB
#define STAGE_MB 2
#endif
    stage_floats = ((size_t)STAGE_MB << 20) / sizeof(float); /* staging buffer */
    stage = xalloc_nozero(stage_floats * sizeof(float));
    LMARK("alloc packed buffer");
    W.wq = malloc(L * sizeof(Mat)); W.wk = malloc(L * sizeof(Mat)); W.wv = malloc(L * sizeof(Mat));
    W.wo = malloc(L * sizeof(Mat)); W.w1 = malloc(L * sizeof(Mat)); W.w3 = malloc(L * sizeof(Mat));
    W.w2 = malloc(L * sizeof(Mat));
    float *dst = packed_buf;
#define PACK(field, off, n_, d_) do { field = load_mat(dst, (off), (n_), (d_)); dst += mat_floats((n_), (d_)); } while (0)
    for (size_t l = 0; l < L; l++) {
        PACK(W.wq[l], wq + l * d * qd, d, qd);
        PACK(W.wk[l], wk + l * d * kv_dim, d, kv_dim);
        PACK(W.wv[l], wv + l * d * kv_dim, d, kv_dim);
        PACK(W.wo[l], wo + l * qd * d, qd, d);
        PACK(W.w1[l], w1 + l * d * h, d, h);
        PACK(W.w3[l], w3 + l * d * h, d, h);
        PACK(W.w2[l], w2 + l * h * d, h, d);
    }
    PACK(W.wcls, wcls, d, cfg.vocab_size);
#undef PACK
    LMARK("read+pack weights");
    free(stage);
    close(fd);

    x = xalloc(d * 4); xb = xalloc(d * 4); xb2 = xalloc(d * 4);
    hb = xalloc(h * 4); hb2 = xalloc(h * 4); q = xalloc(d * 4);
    /* zero-initialised like the reference's calloc; for large sizes calloc
     * hands out fresh zero pages, so only the pages actually used get touched */
    key_cache = calloc(L * cfg.seq_len * kv_dim, sizeof(float));
    value_cache = calloc(L * cfg.seq_len * kv_dim, sizeof(float));
    if (!key_cache || !value_cache) { fprintf(stderr, "alloc failed\n"); exit(1); }
    kcur = xalloc(kv_dim * 4); vcur = xalloc(kv_dim * 4);
    att = xalloc((size_t)cfg.n_heads * cfg.seq_len * 4);
    logits = xalloc((size_t)cfg.vocab_size * 4);

    /* RoPE tables. Same float32 angle as the reference:
     * freq = 1/powf(10000, hd/hs) ; val = pos*freq ; then cos/sin(val). */
    int half = head_size / 2;
    LMARK("alloc+zero activations, KV cache");
    rope_cos = xalloc((size_t)cfg.seq_len * half * 4);
    rope_sin = xalloc((size_t)cfg.seq_len * half * 4);
    const double LN10000 = 9.21034037197618273607;
    for (int j = 0; j < half; j++) {
        int hd = 2 * j;
        float e = hd / (float)head_size;
        float pw = (float)d_exp(LN10000 * (double)e); /* powf(10000, e) */
        float freq = 1.0f / pw;
        for (int pos = 0; pos < cfg.seq_len; pos++) {
            float val = pos * freq;
            double s, c;
            d_sincos((double)val, &s, &c);
            rope_cos[(size_t)pos * half + j] = (float)c;
            rope_sin[(size_t)pos * half + j] = (float)s;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* kernels                                                                    */

static inline float hsum(float32x4_t v) { return vaddvq_f32(v); }

static void rmsnorm(float *o, const float *xx, const float *w, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        float32x4_t v0 = vld1q_f32(xx + j), v1 = vld1q_f32(xx + j + 4);
        a0 = vfmaq_f32(a0, v0, v0); a1 = vfmaq_f32(a1, v1, v1);
    }
    float ss = hsum(vaddq_f32(a0, a1));
    for (; j < n; j++) ss += xx[j] * xx[j];
    ss /= n;
    ss += 1e-5f;
    ss = 1.0f / my_sqrtf(ss);
    float32x4_t s = vdupq_n_f32(ss);
    j = 0;
    for (; j + 4 <= n; j += 4)
        vst1q_f32(o + j, vmulq_f32(vld1q_f32(w + j), vmulq_f32(s, vld1q_f32(xx + j))));
    for (; j < n; j++) o[j] = w[j] * (ss * xx[j]);
}

/* W (d,n) @ x (n,) -> out (d,). Requires n % 16 == 0 for the fast path. */
static void matmul(float *restrict out, const float *restrict xv, const float *restrict w, int n, int d) {
    if (n % 16 != 0) {
        for (int i = 0; i < d; i++) {
            float v = 0.0f;
            for (int j = 0; j < n; j++) v += w[(size_t)i * n + j] * xv[j];
            out[i] = v;
        }
        return;
    }
    /* Two rows per pass, 64-byte loads, 4 accumulators per row. Measured on
     * the M1 as the best weight-streaming pattern (4 or 8 concurrent rows
     * were slower). Memory-bandwidth bound. */
    int i = 0;
    for (; i + 2 <= d; i += 2) {
        const float *r0 = w + (size_t)i * n, *r1 = r0 + n;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int j = 0; j < n; j += 16) {
            float32x4x4_t xx = vld1q_f32_x4(xv + j);
            float32x4x4_t w0 = vld1q_f32_x4(r0 + j);
            float32x4x4_t w1 = vld1q_f32_x4(r1 + j);
            a0 = vfmaq_f32(a0, w0.val[0], xx.val[0]);
            a1 = vfmaq_f32(a1, w0.val[1], xx.val[1]);
            a2 = vfmaq_f32(a2, w0.val[2], xx.val[2]);
            a3 = vfmaq_f32(a3, w0.val[3], xx.val[3]);
            b0 = vfmaq_f32(b0, w1.val[0], xx.val[0]);
            b1 = vfmaq_f32(b1, w1.val[1], xx.val[1]);
            b2 = vfmaq_f32(b2, w1.val[2], xx.val[2]);
            b3 = vfmaq_f32(b3, w1.val[3], xx.val[3]);
        }
        out[i] = hsum(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        out[i + 1] = hsum(vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3)));
    }
    for (; i < d; i++) {
        const float *r = w + (size_t)i * n;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
        for (int j = 0; j < n; j += 8) {
            a0 = vfmaq_f32(a0, vld1q_f32(r + j), vld1q_f32(xv + j));
            a1 = vfmaq_f32(a1, vld1q_f32(r + j + 4), vld1q_f32(xv + j + 4));
        }
        out[i] = hsum(vaddq_f32(a0, a1));
    }
}

/* packed-panel matvec: one sequential weight stream, PR rows at a time */
static void matmul_m(float *restrict out, const float *restrict xv, const Mat *m) {
    if (!m->packed) { matmul(out, xv, m->p, m->n, m->d); return; }
    const float *w = m->p;
    int n = m->n, d = m->d;
    for (int i = 0; i < d; i += PR) {
        float32x4_t a[PR][4];
        for (int r = 0; r < PR; r++) a[r][0] = a[r][1] = a[r][2] = a[r][3] = vdupq_n_f32(0);
        for (int j = 0; j < n; j += 16) {
            float32x4x4_t xx = vld1q_f32_x4(xv + j);
            for (int r = 0; r < PR; r++) {
                float32x4x4_t ww = vld1q_f32_x4(w);
                w += 16;
                a[r][0] = vfmaq_f32(a[r][0], ww.val[0], xx.val[0]);
                a[r][1] = vfmaq_f32(a[r][1], ww.val[1], xx.val[1]);
                a[r][2] = vfmaq_f32(a[r][2], ww.val[2], xx.val[2]);
                a[r][3] = vfmaq_f32(a[r][3], ww.val[3], xx.val[3]);
            }
        }
        for (int r = 0; r < PR; r++)
            out[i + r] = hsum(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3])));
    }
}

/* token embedding row: from the separate table, or gathered back out of the
 * (packed) classifier when the weights are shared */
static void embed(float *dst, int token) {
    int dim = cfg.dim;
    if (!W.shared) { memcpy(dst, W.token_embedding_table + (size_t)token * dim, dim * sizeof(float)); return; }
    const Mat *m = &W.wcls;
    if (!m->packed) { memcpy(dst, m->p + (size_t)token * dim, dim * sizeof(float)); return; }
    const float *base = m->p + (size_t)(token / PR) * PR * dim + (size_t)(token % PR) * 16;
    for (int j = 0; j < dim; j += 16) memcpy(dst + j, base + (size_t)j * PR, 16 * sizeof(float));
}

static void softmax(float *v, int size) {
    float mx = v[0];
    int i = 0;
    if (size >= 8) {
        float32x4_t m0 = vld1q_f32(v), m1 = vld1q_f32(v + 4);
        for (i = 8; i + 8 <= size; i += 8) { m0 = vmaxq_f32(m0, vld1q_f32(v + i)); m1 = vmaxq_f32(m1, vld1q_f32(v + i + 4)); }
        mx = vmaxvq_f32(vmaxq_f32(m0, m1));
    }
    for (; i < size; i++) if (v[i] > mx) mx = v[i];
    float32x4_t m = vdupq_n_f32(mx), acc = vdupq_n_f32(0);
    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t e = vexpq(vsubq_f32(vld1q_f32(v + i), m));
        vst1q_f32(v + i, e);
        acc = vaddq_f32(acc, e);
    }
    float sum = hsum(acc);
    for (; i < size; i++) { v[i] = my_expf(v[i] - mx); sum += v[i]; }
    float32x4_t sv = vdupq_n_f32(sum);
    i = 0;
    for (; i + 4 <= size; i += 4) vst1q_f32(v + i, vdivq_f32(vld1q_f32(v + i), sv));
    for (; i < size; i++) v[i] /= sum;
}

/* a[t] = dot(q, k_t) / sqrt_hs for t < n; keys contiguous (stride head_size) */
static void attn_scores(float *a, const float *qh, const float *kb, int n, float sqrt_hs) {
    int hs = head_size, t = 0;
    if (hs % 16 == 0) {
        float32x4_t div = vdupq_n_f32(sqrt_hs);
        /* 4 positions at once, 4 accumulators each: 16 independent FMA chains */
        for (; t + 4 <= n; t += 4) {
            const float *k0 = kb + (size_t)t * hs, *k1 = k0 + hs, *k2 = k1 + hs, *k3 = k2 + hs;
            float32x4_t s[4][4];
            for (int u = 0; u < 4; u++) s[u][0] = s[u][1] = s[u][2] = s[u][3] = vdupq_n_f32(0);
            for (int i = 0; i < hs; i += 16) {
                float32x4x4_t qq = vld1q_f32_x4(qh + i);
                float32x4x4_t c0 = vld1q_f32_x4(k0 + i), c1 = vld1q_f32_x4(k1 + i);
                float32x4x4_t c2 = vld1q_f32_x4(k2 + i), c3 = vld1q_f32_x4(k3 + i);
                for (int r = 0; r < 4; r++) {
                    s[0][r] = vfmaq_f32(s[0][r], qq.val[r], c0.val[r]);
                    s[1][r] = vfmaq_f32(s[1][r], qq.val[r], c1.val[r]);
                    s[2][r] = vfmaq_f32(s[2][r], qq.val[r], c2.val[r]);
                    s[3][r] = vfmaq_f32(s[3][r], qq.val[r], c3.val[r]);
                }
            }
            float32x4_t r0 = vaddq_f32(vaddq_f32(s[0][0], s[0][1]), vaddq_f32(s[0][2], s[0][3]));
            float32x4_t r1 = vaddq_f32(vaddq_f32(s[1][0], s[1][1]), vaddq_f32(s[1][2], s[1][3]));
            float32x4_t r2 = vaddq_f32(vaddq_f32(s[2][0], s[2][1]), vaddq_f32(s[2][2], s[2][3]));
            float32x4_t r3 = vaddq_f32(vaddq_f32(s[3][0], s[3][1]), vaddq_f32(s[3][2], s[3][3]));
            float32x4_t sc = vpaddq_f32(vpaddq_f32(r0, r1), vpaddq_f32(r2, r3));
            vst1q_f32(a + t, vdivq_f32(sc, div));
        }
        for (; t < n; t++) {
            const float *kt = kb + (size_t)t * hs;
            float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
            for (int i = 0; i < hs; i += 16) {
                float32x4x4_t qq = vld1q_f32_x4(qh + i), c = vld1q_f32_x4(kt + i);
                s0 = vfmaq_f32(s0, qq.val[0], c.val[0]); s1 = vfmaq_f32(s1, qq.val[1], c.val[1]);
                s2 = vfmaq_f32(s2, qq.val[2], c.val[2]); s3 = vfmaq_f32(s3, qq.val[3], c.val[3]);
            }
            a[t] = hsum(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3))) / sqrt_hs;
        }
        return;
    }
    for (; t < n; t++) {
        const float *kt = kb + (size_t)t * hs;
        float s = 0.0f;
        for (int i = 0; i < hs; i++) s += qh[i] * kt[i];
        a[t] = s / sqrt_hs;
    }
}

/* o = sum_t a[t] * v_t for t < n; values contiguous (stride head_size) */
static void attn_values(float *o, const float *a, const float *vb, int n) {
    int hs = head_size;
    if (hs % 64 == 0) {
        for (int c = 0; c < hs; c += 64) {          /* 64-float chunk: 16 accumulators */
            float32x4_t acc[16];
            for (int i = 0; i < 16; i++) acc[i] = vdupq_n_f32(0);
            for (int t = 0; t < n; t++) {
                const float *vt = vb + (size_t)t * hs + c;
                float32x4_t av = vdupq_n_f32(a[t]);
                for (int i = 0; i < 16; i += 4) {
                    float32x4x4_t vv = vld1q_f32_x4(vt + 4 * i);
                    acc[i] = vfmaq_f32(acc[i], av, vv.val[0]);
                    acc[i + 1] = vfmaq_f32(acc[i + 1], av, vv.val[1]);
                    acc[i + 2] = vfmaq_f32(acc[i + 2], av, vv.val[2]);
                    acc[i + 3] = vfmaq_f32(acc[i + 3], av, vv.val[3]);
                }
            }
            for (int i = 0; i < 16; i++) vst1q_f32(o + c + 4 * i, acc[i]);
        }
        return;
    }
    if (hs % 16 == 0) {
        for (int c = 0; c < hs; c += 16) {
            float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
            for (int t = 0; t < n; t++) {
                float32x4x4_t vv = vld1q_f32_x4(vb + (size_t)t * hs + c);
                float32x4_t av = vdupq_n_f32(a[t]);
                a0 = vfmaq_f32(a0, av, vv.val[0]); a1 = vfmaq_f32(a1, av, vv.val[1]);
                a2 = vfmaq_f32(a2, av, vv.val[2]); a3 = vfmaq_f32(a3, av, vv.val[3]);
            }
            vst1q_f32(o + c, a0); vst1q_f32(o + c + 4, a1); vst1q_f32(o + c + 8, a2); vst1q_f32(o + c + 12, a3);
        }
        return;
    }
    memset(o, 0, hs * sizeof(float));
    for (int t = 0; t < n; t++)
        for (int i = 0; i < hs; i++) o[i] += a[t] * vb[(size_t)t * hs + i];
}

static float *forward(int token, int pos) {
    int dim = cfg.dim, hidden = cfg.hidden_dim, half = head_size / 2;
    embed(x, token);
    const float *rc = rope_cos + (size_t)pos * half, *rs = rope_sin + (size_t)pos * half;
    float sqrt_hs = my_sqrtf((float)head_size);
#ifdef PROFILE
    prof_forwards++;
#endif
    PSTART();

    for (size_t l = 0; l < (size_t)cfg.n_layers; l++) {
        rmsnorm(xb, x, W.rms_att_weight + l * dim, dim);
        float *k = kcur, *v = vcur;
        PMARK(P_RMS);
        matmul_m(q, xb, &W.wq[l]);
        matmul_m(k, xb, &W.wk[l]);
        matmul_m(v, xb, &W.wv[l]);
        PMARK(P_QKV);

        /* RoPE */
        for (int i = 0; i < dim; i += 2) {
            int j = (i % head_size) / 2;
            float fcr = rc[j], fci = rs[j];
            float v0 = q[i], v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
            if (i < kv_dim) {
                v0 = k[i]; v1 = k[i + 1];
                k[i] = v0 * fcr - v1 * fci;
                k[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        /* store this position into the cache */
        for (int kh = 0; kh < cfg.n_kv_heads; kh++) {
            memcpy(key_cache + kv_index(l, kh, pos), k + kh * head_size, head_size * sizeof(float));
            memcpy(value_cache + kv_index(l, kh, pos), v + kh * head_size, head_size * sizeof(float));
        }

        PMARK(P_ROPE);
        /* attention */
        for (int h = 0; h < cfg.n_heads; h++) {
            const float *qh = q + h * head_size;
            float *a = att + (size_t)h * cfg.seq_len;
            const float *kb = key_cache + kv_index(l, h / kv_mul, 0);
            const float *vb = value_cache + kv_index(l, h / kv_mul, 0);
            attn_scores(a, qh, kb, pos + 1, sqrt_hs);
            softmax(a, pos + 1);
            attn_values(xb + h * head_size, a, vb, pos + 1);
        }

        PMARK(P_ATT);
        matmul_m(xb2, xb, &W.wo[l]);
        PMARK(P_WO);
        for (int i = 0; i < dim; i++) x[i] += xb2[i];
        PMARK(P_RESID);

        rmsnorm(xb, x, W.rms_ffn_weight + l * dim, dim);
        PMARK(P_RMS);
        matmul_m(hb, xb, &W.w1[l]);
        matmul_m(hb2, xb, &W.w3[l]);
        PMARK(P_W13);

        /* SwiGLU: silu(hb) * hb2 */
        int i = 0;
        for (; i + 4 <= hidden; i += 4) {
            float32x4_t hv = vld1q_f32(hb + i);
            float32x4_t sig = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), vexpq(vnegq_f32(hv))));
            vst1q_f32(hb + i, vmulq_f32(vmulq_f32(hv, sig), vld1q_f32(hb2 + i)));
        }
        for (; i < hidden; i++) {
            float val = hb[i];
            val *= 1.0f / (1.0f + my_expf(-val));
            hb[i] = val * hb2[i];
        }

        PMARK(P_SILU);
        matmul_m(xb, hb, &W.w2[l]);
        PMARK(P_W2);
        for (int j = 0; j < dim; j++) x[j] += xb[j];
        PMARK(P_RESID);
    }
    rmsnorm(x, x, W.rms_final_weight, dim);
    PMARK(P_RMS);
    matmul_m(logits, x, &W.wcls);
    PMARK(P_CLS);
    return logits;
}

/* Greedy argmax, lowest index wins ties (same result as the reference's
 * strict '>' scan). Vector pass finds the max value, then the first index
 * holding it. If any NaN is present, fall back to the reference scan. */
static int argmax(const float *v, int n) {
    PSTART();
    int mi = 0;
    if (n >= 16) {
        float32x4_t m0 = vld1q_f32(v), m1 = vld1q_f32(v + 4);
        uint32x4_t ok = vandq_u32(vceqq_f32(m0, m0), vceqq_f32(m1, m1));
        int i = 8;
        for (; i + 8 <= n; i += 8) {
            float32x4_t a = vld1q_f32(v + i), b = vld1q_f32(v + i + 4);
            ok = vandq_u32(ok, vandq_u32(vceqq_f32(a, a), vceqq_f32(b, b)));
            m0 = vmaxq_f32(m0, a); m1 = vmaxq_f32(m1, b);
        }
        float mv = vmaxvq_f32(vmaxq_f32(m0, m1));
        int nan = vminvq_u32(ok) == 0;
        for (int j = i; j < n; j++) { if (v[j] != v[j]) nan = 1; else if (v[j] > mv) mv = v[j]; }
        if (!nan) {
            while (v[mi] != mv) mi++;
            PMARK(P_ARGMAX);
            return mi;
        }
    }
    float mv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mv) { mv = v[i]; mi = i; }
    PMARK(P_ARGMAX);
    return mi;
}

/* ------------------------------------------------------------------------- */
/* tokenizer (same algorithm as reference; hash lookup instead of bsearch)    */

static char **vocab;
static int *vocab_len;
static float *vocab_scores;
static unsigned max_token_length;
static unsigned char byte_pieces[512];
static int *htab; static unsigned hmask;
static char *sbuf, *pbuf;

static uint32_t hash_bytes(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}
/* lookup of a NUL-terminated-equivalent string (s, n); -1 if absent.
 * Vocab entries are compared as C strings, like strcmp in the reference. */
static int lookup(const char *s, size_t n) {
    uint32_t h = hash_bytes(s, n) & hmask;
    while (htab[h] >= 0) {
        int id = htab[h];
        if ((size_t)vocab_len[id] == n && memcmp(vocab[id], s, n) == 0) return id;
        h = (h + 1) & hmask;
    }
    return -1;
}

static void load_tokenizer(const char *path, int vs) {
    vocab = malloc(vs * sizeof(char *));
    vocab_len = malloc(vs * sizeof(int));
    vocab_scores = malloc(vs * sizeof(float));
    for (int i = 0; i < 256; i++) { byte_pieces[i * 2] = (unsigned char)i; byte_pieces[i * 2 + 1] = 0; }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "couldn't load %s\n", path); exit(1); }
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
    for (int i = 0; i < vs; i++) {
        int len;
        if (fread(vocab_scores + i, sizeof(float), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        if (fread(&len, sizeof(int), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        vocab[i] = malloc(len + 1);
        if (len > 0 && fread(vocab[i], len, 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        vocab[i][len] = 0;
        vocab_len[i] = (int)strlen(vocab[i]); /* C-string semantics as in reference */
    }
    fclose(f);
    unsigned cap = 1;
    while (cap < (unsigned)vs * 2) cap <<= 1;
    hmask = cap - 1;
    htab = malloc(cap * sizeof(int));
    for (unsigned i = 0; i < cap; i++) htab[i] = -1;
    /* the reference's bsearch over a qsort'ed array returns *some* entry with
     * an equal string; for duplicates we keep the first id inserted. */
    for (int i = 0; i < vs; i++) {
        if (lookup(vocab[i], vocab_len[i]) >= 0) continue;
        uint32_t h = hash_bytes(vocab[i], vocab_len[i]) & hmask;
        while (htab[h] >= 0) h = (h + 1) & hmask;
        htab[h] = i;
    }
    sbuf = malloc(max_token_length * 2 + 3 + 8);
    int maxlen = 0;
    for (int i = 0; i < vs; i++) if (vocab_len[i] > maxlen) maxlen = vocab_len[i];
    pbuf = malloc(2 * (size_t)maxlen + 8);
}

static int pair_id(int a, int b) {
    size_t la = vocab_len[a], lb = vocab_len[b];
    memcpy(pbuf, vocab[a], la);
    memcpy(pbuf + la, vocab[b], lb);
    return lookup(pbuf, la + lb);
}

/* tokens must hold strlen(text)+3 entries */
static int encode(const char *text, int *tokens) {
    PSTART();
    int n = 0;
    tokens[n++] = 1; /* BOS */
    if (text[0] != 0) tokens[n++] = lookup(" ", 1);
    size_t str_len = 0;
    for (const char *c = text; *c != 0; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        sbuf[str_len++] = *c;
        sbuf[str_len] = 0;
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = lookup(sbuf, str_len);
        if (id != -1) tokens[n++] = id;
        else for (size_t i = 0; i < str_len; i++) tokens[n++] = (unsigned char)sbuf[i] + 3;
        str_len = 0;
    }
    if (n < 2) { PMARK(P_ENCODE); return n; }
    /* incremental merge: pid[i] = vocab id of merge(tokens[i], tokens[i+1]) */
    int *pid = malloc(n * sizeof(int));
    for (int i = 0; i + 1 < n; i++) pid[i] = pair_id(tokens[i], tokens[i + 1]);
    while (1) {
        float best = -1e10f;
        int best_idx = -1;
        for (int i = 0; i + 1 < n; i++) {
            int id = pid[i];
            if (id != -1 && vocab_scores[id] > best) { best = vocab_scores[id]; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = pid[best_idx];
        for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
        for (int i = best_idx + 1; i < n - 2; i++) pid[i] = pid[i + 1];
        n--;
        if (best_idx > 0) pid[best_idx - 1] = pair_id(tokens[best_idx - 1], tokens[best_idx]);
        if (best_idx + 1 < n) pid[best_idx] = pair_id(tokens[best_idx], tokens[best_idx + 1]);
    }
    free(pid);
    PMARK(P_ENCODE);
    return n;
}

static const char *decode(int prev, int token) {
    const char *piece = vocab[token];
    if (prev == 1 && piece[0] == ' ') piece++;
    unsigned char bv;
    if (sscanf(piece, "<0x%02hhX>", &bv) == 1) piece = (const char *)byte_pieces + bv * 2;
    return piece;
}

/* ------------------------------------------------------------------------- */
/* protocol                                                                   */

static int parse_ids(const char *s, int **out, int *cap) {
    int n = 0;
    char *end;
    while (1) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        s = end;
        if (n == *cap) { *cap = *cap ? *cap * 2 : 1024; *out = realloc(*out, *cap * sizeof(int)); }
        (*out)[n++] = (int)v;
    }
    return n;
}

static int valid_id(int t) { return t >= 0 && t < cfg.vocab_size; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tokenizer.bin\n", argv[0]); return 1; }
    load_model(argv[1]);
    LMARK("rope tables");
    load_tokenizer(argv[2], cfg.vocab_size);
    LMARK("tokenizer");
    static char outbuf[1 << 16];
    setvbuf(stdout, outbuf, _IOFBF, sizeof outbuf);
    printf("READY\n"); fflush(stdout);

    char *line = NULL; size_t lcap = 0; ssize_t len;
    int *ids = NULL, idcap = 0;
    while ((len = getline(&line, &lcap, stdin)) >= 0) {
        if (len > 0 && line[len - 1] == '\n') line[--len] = 0;
        if (strncmp(line, "QUIT", 4) == 0 && (line[4] == 0 || line[4] == ' ')) break;
        if (strncmp(line, "ENCODE", 6) == 0 && (line[6] == 0 || line[6] == ' ')) {
            const char *text = line[6] ? line + 7 : "";
            int *toks = malloc((strlen(text) + 3) * sizeof(int));
            int n = encode(text, toks);
            printf("IDS");
            for (int i = 0; i < n; i++) printf(" %d", toks[i]);
            printf("\n"); fflush(stdout);
            free(toks);
        } else if (strncmp(line, "PIECE ", 6) == 0) {
            int prev = 0, id = 0;
            sscanf(line + 6, "%d %d", &prev, &id);
            printf("HEX ");
            if (valid_id(id)) {
                const unsigned char *pc = (const unsigned char *)decode(prev, id);
                for (; *pc; pc++) printf("%02x", *pc);
            }
            printf("\n"); fflush(stdout);
        } else if (strncmp(line, "GENERATE ", 9) == 0) {
            char *end;
            long max_new = strtol(line + 9, &end, 10);
            const char *text = end;
            if (*text == ' ') text++;
            int *toks = malloc((strlen(text) + 3) * sizeof(int));
            int n = encode(text, toks);
            int count = 0;
            const char *why = "max";
            if (max_new > 0) {
                int token = toks[0], pos = 0;
                while (1) {
                    if (pos >= cfg.seq_len) { why = "context"; break; }
                    float *lg = forward(token, pos);
                    if (pos < n - 1) { token = toks[pos + 1]; pos++; continue; }
                    int next = argmax(lg, cfg.vocab_size);
                    pos++;
                    if (next == 1) { why = "stop"; break; }
                    printf("T %d\n", next); fflush(stdout);
                    count++;
                    if (count >= max_new) { why = "max"; break; }
                    token = next;
                }
            }
            printf("DONE %d %s\n", count, why); fflush(stdout);
            free(toks);
        } else if (strncmp(line, "FIXED", 5) == 0 && (line[5] == 0 || line[5] == ' ')) {
            int n = parse_ids(line + 5, &ids, &idcap);
            if (n > cfg.seq_len) n = cfg.seq_len;
            int *am = malloc((n + 1) * sizeof(int));
            for (int p = 0; p < n; p++) {
                int t = valid_id(ids[p]) ? ids[p] : 0;
                am[p] = argmax(forward(t, p), cfg.vocab_size);
            }
            printf("DONE %d", n);
            for (int p = 0; p < n; p++) printf(" %d", am[p]);
            printf("\n"); fflush(stdout);
            free(am);
        } else if (strncmp(line, "LOGITS ", 7) == 0) {
            char *path = line + 7;
            char *sp = strchr(path, ' ');
            const char *rest = "";
            if (sp) { *sp = 0; rest = sp + 1; }
            int n = parse_ids(rest, &ids, &idcap);
            if (n > cfg.seq_len) n = cfg.seq_len;
            FILE *f = fopen(path, "wb");
            if (!f) { printf("ERROR cannot open %s\n", path); fflush(stdout); continue; }
            for (int p = 0; p < n; p++) {
                int t = valid_id(ids[p]) ? ids[p] : 0;
                float *lg = forward(t, p);
                PSTART();
                fwrite(lg, sizeof(float), cfg.vocab_size, f);
                PMARK(P_LOGIO);
            }
            PSTART();
            fclose(f);
            PMARK(P_LOGIO);
            printf("DONE %d\n", n); fflush(stdout);
        } else {
            printf("ERROR unknown command\n"); fflush(stdout);
        }
    }
    free(line);
#ifdef PROFILE
    prof_report();
#endif
    return 0;
}
