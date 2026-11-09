/* CB2 port of reference/run.c (llama2.c): C + Apple Accelerate, float32.
 * Interface: see INTERFACE.md. Matrix work goes through Accelerate (cblas_sgemm / cblas_sgemv);
 * element-wise vector work uses vDSP/vForce or plain scalar C.
 */
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

/* Optional section profiler: build with -DPROF; prints a per-section breakdown to stderr on exit. */
#ifdef PROF
#include <mach/mach_time.h>
static double prof[16]; static uint64_t pt0; static long pcalls;
#define P0() (pt0 = mach_absolute_time())
#define P(i) do { uint64_t _t = mach_absolute_time(); prof[i] += (double)(_t - pt0); pt0 = _t; } while (0)
#else
#define P0() ((void)0)
#define P(i) ((void)0)
#endif

/* ---------------------------------------------------------------- model */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

static Config cfg;
static int dim, hid, nl, nh, nkv, V, S, hs, kvd, kv_mul, shared_cls;

/* A weight matrix W (d x n, PyTorch layout: y = W x) stored transposed as consecutive column blocks:
 * block b holds W[j0:j0+w, :]^T as a contiguous row-major (n x w) array starting at data + n*j0.
 * Block widths are at most BLOCK_MAX (256) columns, where Accelerate's transposed sgemv streams fastest
 * on this machine (see NOTES.md, profiling). */
typedef struct { int n, d, nb; int *j0, *w; float *data; } Mat;

#ifndef BLOCK_MAX
#define BLOCK_MAX 256
#endif
static void mat_plan(Mat *m, int n, int d, float *data) {
    m->n = n; m->d = d; m->data = data;
    m->nb = (d + BLOCK_MAX - 1) / BLOCK_MAX;
    m->j0 = malloc(m->nb * sizeof(int)); m->w = malloc(m->nb * sizeof(int));
    int units = d / 16, rem = d % 16, j = 0;
    for (int b = 0; b < m->nb; b++) {
        int w = 16 * (units / m->nb + (b < units % m->nb ? 1 : 0));
        if (b == m->nb - 1) w += rem;
        m->j0[b] = j; m->w[b] = w; j += w;
    }
}
/* fill from row-major W (d x n) */
static void mat_fill(Mat *m, const float *W) {
    for (int b = 0; b < m->nb; b++)
        vDSP_mtrans(W + (size_t)m->j0[b] * m->n, 1, m->data + (size_t)m->n * m->j0[b], 1, m->n, m->w[b]);
}

static float *emb, *rms_att, *rms_ffn, *rms_final; /* emb rows kept only if the classifier is unshared */
static Mat cls, *wq, *wk, *wv, *wo, *w1, *w3, *w2;
static float *kcache, *vcache;          /* (layer, seq_len, kv_dim) */
static float *rope_cos, *rope_sin;      /* (seq_len, head_size/2) */

#define CHUNK 512   /* max positions forwarded together */
#define ATT_ROWS 64 /* query rows per causal attention group in chunked forward */
#define ATT_BLOCKDIAG_MIN 160 /* single-token attention switches to the all-heads sgemm path at this many keys */
static float *bx, *bxb, *bxb2, *bq, *bh1, *bh3, *btmp, *batt, *blogits;
static float *qblk, *bsc, *bho; /* single-token attention: block-diagonal queries, scores, head outputs */

static void *xalloc(size_t n) {
    void *p = NULL;
    if (n == 0) n = 1;
    if (posix_memalign(&p, 16384, n) != 0 || !p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}

static void pread_all(int fd, void *dst, size_t n, off_t off) {
    char *d = dst;
    while (n > 0) {
        ssize_t r = pread(fd, d, n > (1u << 30) ? (1u << 30) : n, off);
        if (r <= 0) { fprintf(stderr, "read failed\n"); exit(1); }
        d += r; n -= (size_t)r; off += r;
    }
}

static void load_model(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    pread_all(fd, &cfg, sizeof(Config), 0);
    shared_cls = cfg.vocab_size > 0;
    cfg.vocab_size = abs(cfg.vocab_size);
    dim = cfg.dim; hid = cfg.hidden_dim; nl = cfg.n_layers; nh = cfg.n_heads; nkv = cfg.n_kv_heads;
    V = cfg.vocab_size; S = cfg.seq_len;
    hs = dim / nh; kvd = dim * nkv / nh; kv_mul = nh / nkv;

    /* file offsets (in floats, after the header) */
    size_t L = (size_t)nl, D = dim, H = hid, KV = kvd, VV = V;
    size_t o_emb = 0;
    size_t o_rmsatt = o_emb + VV * D;
    size_t o_wq = o_rmsatt + L * D;
    size_t o_wk = o_wq + L * D * D;
    size_t o_wv = o_wk + L * D * KV;
    size_t o_wo = o_wv + L * D * KV;
    size_t o_rmsffn = o_wo + L * D * D;
    size_t o_w1 = o_rmsffn + L * D;
    size_t o_w2 = o_w1 + L * D * H;
    size_t o_w3 = o_w2 + L * H * D;
    size_t o_rmsfin = o_w3 + L * D * H;
    size_t o_cls = o_rmsfin + D + (size_t)S * hs; /* skip legacy freq_cis_real/imag */
#define OFF(x) ((off_t)sizeof(Config) + (off_t)(x) * 4)

    rms_att = xalloc(L * D * 4);
    pread_all(fd, rms_att, L * D * 4, OFF(o_rmsatt));
    rms_ffn = xalloc(L * D * 4);
    pread_all(fd, rms_ffn, L * D * 4, OFF(o_rmsffn));
    rms_final = xalloc(D * 4);
    pread_all(fd, rms_final, D * 4, OFF(o_rmsfin));

#ifdef LOAD_PREAD
    /* staging buffer for one row-major matrix before it is transposed into blocks */
    size_t tmpn = VV * D;
    if (H * D > tmpn) tmpn = H * D;
    float *tmp = xalloc(tmpn * 4);
#define SRC(off, cnt) (pread_all(fd, tmp, (size_t)(cnt) * 4, OFF(off)), (const float *)tmp)
#else
    /* transpose straight out of a read-only mapping of the checkpoint (one pass over the file) */
    off_t fsize = lseek(fd, 0, SEEK_END);
    char *map = mmap(NULL, (size_t)fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }
    if ((size_t)OFF(o_cls + (shared_cls ? 0 : VV * D)) > (size_t)fsize) { fprintf(stderr, "checkpoint too small\n"); exit(1); }
#define SRC(off, cnt) ((const float *)(map + OFF(off)))
#endif

    /* classifier (and, when shared, the embedding table: rows are gathered from the blocks) */
    mat_plan(&cls, dim, V, xalloc(VV * D * 4));
    if (!shared_cls) {
        emb = xalloc(VV * D * 4);
        pread_all(fd, emb, VV * D * 4, OFF(o_emb));
    }
    mat_fill(&cls, SRC(shared_cls ? o_emb : o_cls, VV * D));

    /* per-layer storage, one contiguous region per layer in use order: wq, wk, wv, wo, w1, w3, w2 */
    size_t per = D * D + 2 * KV * D + D * D + 2 * H * D + D * H;
    float *blk = xalloc(L * per * 4);
    wq = malloc(L * sizeof(Mat)); wk = malloc(L * sizeof(Mat)); wv = malloc(L * sizeof(Mat));
    wo = malloc(L * sizeof(Mat)); w1 = malloc(L * sizeof(Mat)); w3 = malloc(L * sizeof(Mat));
    w2 = malloc(L * sizeof(Mat));
    struct { Mat *m; int n, d; size_t off, stride; } spec[7] = {
        {wq, dim, dim, o_wq, D * D}, {wk, dim, kvd, o_wk, KV * D}, {wv, dim, kvd, o_wv, KV * D},
        {wo, dim, dim, o_wo, D * D}, {w1, dim, hid, o_w1, H * D}, {w3, dim, hid, o_w3, H * D},
        {w2, hid, dim, o_w2, D * H}};
    for (size_t l = 0; l < L; l++) {
        float *b = blk + l * per;
        for (int k = 0; k < 7; k++) {
            Mat *m = &spec[k].m[l];
            mat_plan(m, spec[k].n, spec[k].d, b);
            mat_fill(m, SRC(spec[k].off + l * spec[k].stride, spec[k].stride));
            b += spec[k].stride;
        }
    }
#ifdef LOAD_PREAD
    free(tmp);
#else
    munmap(map, (size_t)fsize);
#endif
    close(fd);

    kcache = xalloc(L * S * KV * 4);
    vcache = xalloc(L * S * KV * 4);
    memset(kcache, 0, L * S * KV * 4);
    memset(vcache, 0, L * S * KV * 4);

    /* RoPE table: identical float expressions to the reference, evaluated once per (pos, pair) */
    int half = hs / 2;
    rope_cos = xalloc((size_t)S * half * 4);
    rope_sin = xalloc((size_t)S * half * 4);
    for (int pos = 0; pos < S; pos++)
        for (int j = 0; j < half; j++) {
            int head_dim = 2 * j;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)hs);
            float val = pos * freq;
            rope_cos[(size_t)pos * half + j] = cosf(val);
            rope_sin[(size_t)pos * half + j] = sinf(val);
        }

    size_t C = CHUNK;
    bx = xalloc(C * D * 4); bxb = xalloc(C * D * 4); bxb2 = xalloc(C * D * 4);
    bq = xalloc(C * D * 4); bh1 = xalloc(C * H * 4); bh3 = xalloc(C * H * 4); btmp = xalloc(C * H * 4);
    batt = xalloc(C * (size_t)S * 4); blogits = xalloc(C * VV * 4);
    qblk = xalloc((size_t)kvd * nh * 4); memset(qblk, 0, (size_t)kvd * nh * 4); /* off-block entries stay 0 */
    bsc = xalloc((size_t)S * nh * 4); bho = xalloc((size_t)nh * kvd * 4);
}

/* ------------------------------------------------------------ forward */

/* y (c x d, row stride ldy) = x (c x n) @ W^T, one Accelerate call per column block of the transposed weights */
static void matmul(float *y, int ldy, const float *x, const Mat *m, int c) {
    const int n = m->n;
    for (int b = 0; b < m->nb; b++) {
        const float *blk = m->data + (size_t)n * m->j0[b];
        const int w = m->w[b];
        if (c == 1)
            cblas_sgemv(CblasRowMajor, CblasTrans, n, w, 1.0f, blk, w, x, 1, 0.0f, y + m->j0[b], 1);
        else
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, w, n, 1.0f, x, n, blk, w, 0.0f,
                        y + m->j0[b], ldy);
    }
}

/* embedding row for token t */
static void embed(float *x, int t) {
    if (!shared_cls) { memcpy(x, emb + (size_t)t * dim, dim * 4); return; }
    int b = 0;
    while (b + 1 < cls.nb && cls.j0[b + 1] <= t) b++;
    const float *blk = cls.data + (size_t)dim * cls.j0[b];
    cblas_scopy(dim, blk + (t - cls.j0[b]), cls.w[b], x, 1);
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss;
    vDSP_svesq(x, 1, &ss, n);
    ss /= n;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    vDSP_vsmul(x, 1, &ss, o, 1, n);
    vDSP_vmul(w, 1, o, 1, o, 1, n);
}

static void softmax(float *x, int n) {
    float mx, sum;
    vDSP_maxv(x, 1, &mx, n);
    float neg = -mx;
    vDSP_vsadd(x, 1, &neg, x, 1, n);
    vvexpf(x, x, &n);
    vDSP_sve(x, 1, &sum, n);
    vDSP_vsdiv(x, 1, &sum, x, 1, n);
}

static void rope(float *vec, int n, int pos) {
    int half = hs / 2;
    const float *cs = rope_cos + (size_t)pos * half, *sn = rope_sin + (size_t)pos * half;
    for (int i = 0; i < n; i += 2) {
        int j = (i % hs) / 2;
        float fcr = cs[j], fci = sn[j];
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

/* Forward c tokens at positions p0..p0+c-1 (c <= CHUNK); logits for each go to blogits (c x V). */
static void forward_chunk(const int *toks, int p0, int c) {
    const float sq = sqrtf((float)hs);
    const int T = p0 + c; /* keys visible to the last row */
    P0();
    for (int r = 0; r < c; r++) embed(bx + (size_t)r * dim, toks[r]);

    for (int l = 0; l < nl; l++) {
        float *kc = kcache + (size_t)l * S * kvd, *vc = vcache + (size_t)l * S * kvd;
        float *kp = kc + (size_t)p0 * kvd, *vp = vc + (size_t)p0 * kvd; /* this chunk's cache rows */
        for (int r = 0; r < c; r++) rmsnorm(bxb + (size_t)r * dim, bx + (size_t)r * dim, rms_att + (size_t)l * dim, dim);
        P(0);
        matmul(bq, dim, bxb, &wq[l], c);
        matmul(kp, kvd, bxb, &wk[l], c); /* k and v go straight into the KV cache */
        matmul(vp, kvd, bxb, &wv[l], c);
        P(1);
        for (int r = 0; r < c; r++) {
            rope(bq + (size_t)r * dim, dim, p0 + r);
            rope(kp + (size_t)r * kvd, kvd, p0 + r);
        }
        P(2);
        if (c == 1 && T < ATT_BLOCKDIAG_MIN) {
            /* short context: per-head sgemv over the strided cache rows */
            for (int h = 0; h < nh; h++) {
                const float *kh = kc + (h / kv_mul) * hs, *vh = vc + (h / kv_mul) * hs;
                cblas_sgemv(CblasRowMajor, CblasNoTrans, T, hs, 1.0f, kh, kvd, bq + h * hs, 1, 0.0f, batt, 1);
                vDSP_vsdiv(batt, 1, &sq, batt, 1, T);
                softmax(batt, T);
                cblas_sgemv(CblasRowMajor, CblasTrans, T, hs, 1.0f, vh, kvd, batt, 1, 0.0f, bxb + h * hs, 1);
            }
        } else if (c == 1) {
            /* long context, all heads at once: column h of qblk (kv_dim x n_heads) holds q_h at its kv head's rows, zeros
             * elsewhere, so K (T x kv_dim) @ qblk gives every head's scores while reading K once, contiguously.
             * att (n_heads x T) @ V (T x kv_dim) then holds each head's output in its diagonal block. */
            for (int h = 0; h < nh; h++)
                cblas_scopy(hs, bq + h * hs, 1, qblk + (size_t)(h / kv_mul) * hs * nh + h, nh);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, nh, kvd, 1.0f, kc, kvd, qblk, nh, 0.0f,
                        bsc, nh);
            vDSP_mtrans(bsc, 1, batt, 1, nh, T);
            for (int h = 0; h < nh; h++) {
                float *row = batt + (size_t)h * T;
                vDSP_vsdiv(row, 1, &sq, row, 1, T);
                softmax(row, T);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nh, kvd, T, 1.0f, batt, T, vc, kvd, 0.0f,
                        bho, kvd);
            for (int h = 0; h < nh; h++)
                memcpy(bxb + h * hs, bho + (size_t)h * kvd + (h / kv_mul) * hs, hs * 4);
        } else for (int h = 0; h < nh; h++) {
            const float *qh = bq + h * hs;
            const float *kh = kc + (h / kv_mul) * hs, *vh = vc + (h / kv_mul) * hs;
            float *oh = bxb + h * hs;
            /* causal attention in row groups: rows r0..r1-1 only need keys 0..p0+r1-1 */
            for (int r0 = 0; r0 < c; r0 += ATT_ROWS) {
                int r1 = r0 + ATT_ROWS < c ? r0 + ATT_ROWS : c, m = r1 - r0, Tk = p0 + r1;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, Tk, hs, 1.0f, qh + (size_t)r0 * dim, dim,
                            kh, kvd, 0.0f, batt, Tk);
                for (int r = r0; r < r1; r++) {
                    float *row = batt + (size_t)(r - r0) * Tk;
                    int n = p0 + r + 1; /* causal: keys 0..pos */
                    vDSP_vsdiv(row, 1, &sq, row, 1, n);
                    softmax(row, n);
                    if (n < Tk) memset(row + n, 0, (size_t)(Tk - n) * 4);
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, hs, Tk, 1.0f, batt, Tk, vh, kvd,
                            0.0f, oh + (size_t)r0 * dim, dim);
            }
        }
        P(3); matmul(bxb2, dim, bxb, &wo[l], c); P(4);
        vDSP_vadd(bx, 1, bxb2, 1, bx, 1, (vDSP_Length)c * dim);

        for (int r = 0; r < c; r++) rmsnorm(bxb + (size_t)r * dim, bx + (size_t)r * dim, rms_ffn + (size_t)l * dim, dim);
        P(0);
        matmul(bh1, hid, bxb, &w1[l], c);
        matmul(bh3, hid, bxb, &w3[l], c);
        P(5);
        /* SwiGLU, same operation order as the reference: val = h1 * (1 / (1 + exp(-h1))); val *= h3 */
        {
            int n = c * hid;
            const float one = 1.0f;
            vDSP_vneg(bh1, 1, btmp, 1, n);
            vvexpf(btmp, btmp, &n);
            vDSP_vsadd(btmp, 1, &one, btmp, 1, n);
            vDSP_svdiv(&one, btmp, 1, btmp, 1, n);
            vDSP_vmul(bh1, 1, btmp, 1, btmp, 1, n);
            vDSP_vmul(btmp, 1, bh3, 1, bh1, 1, n);
        }
        P(6); matmul(bxb, dim, bh1, &w2[l], c); P(7);
        vDSP_vadd(bx, 1, bxb, 1, bx, 1, (vDSP_Length)c * dim);
    }
    for (int r = 0; r < c; r++) rmsnorm(bx + (size_t)r * dim, bx + (size_t)r * dim, rms_final, dim);
    P(0); matmul(blogits, V, bx, &cls, c); P(8);
#ifdef PROF
    pcalls++;
#endif
}

static int argmax(const float *x, int n) {
    int mi = 0;
    float mv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > mv) { mi = i; mv = x[i]; }
    return mi;
}

/* ---------------------------------------------------------- tokenizer */

typedef struct { char *str; int id; } TokenIndex;
static char **vocab;
static float *vocab_scores;
static TokenIndex *sorted_vocab;
static unsigned int max_token_length;
static unsigned char byte_pieces[512];

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

static void load_tokenizer(const char *path) {
    vocab = malloc((size_t)V * sizeof(char *));
    vocab_scores = malloc((size_t)V * sizeof(float));
    for (int i = 0; i < 256; i++) { byte_pieces[i * 2] = (unsigned char)i; byte_pieces[i * 2 + 1] = 0; }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "couldn't load %s\n", path); exit(1); }
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) exit(1);
    int len;
    for (int i = 0; i < V; i++) {
        if (fread(vocab_scores + i, sizeof(float), 1, f) != 1) exit(1);
        if (fread(&len, sizeof(int), 1, f) != 1) exit(1);
        vocab[i] = malloc(len + 1);
        if (len > 0 && fread(vocab[i], len, 1, f) != 1) exit(1);
        vocab[i][len] = 0;
    }
    fclose(f);
    sorted_vocab = malloc((size_t)V * sizeof(TokenIndex));
    for (int i = 0; i < V; i++) { sorted_vocab[i].str = vocab[i]; sorted_vocab[i].id = i; }
    qsort(sorted_vocab, V, sizeof(TokenIndex), compare_tokens);
}

static int str_lookup(char *str) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, V, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

/* Same algorithm as the reference encode(), with bos=1, eos=0. tokens needs strlen(text)+3 slots. */
static void encode(const char *text, int *tokens, int *n_tokens) {
    char *buf = malloc(max_token_length * 2 + 1 + 2);
    size_t str_len = 0;
    *n_tokens = 0;
    tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') tokens[(*n_tokens)++] = str_lookup(" ");
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        buf[str_len++] = *c;
        buf[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(buf);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else for (size_t i = 0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3;
        str_len = 0;
    }
    /* Merge loop. Same result as the reference (each step merges the highest-scoring adjacent pair, lowest
     * index on ties), but each pair's candidate merge is cached and only the two pairs next to a merge are
     * looked up again, instead of re-running sprintf + bsearch for every pair on every step. */
    int n = *n_tokens;
    int *cid = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    float *csc = malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
#define PAIR(i) do { sprintf(buf, "%s%s", vocab[tokens[i]], vocab[tokens[(i) + 1]]); \
                     cid[i] = str_lookup(buf); csc[i] = cid[i] != -1 ? vocab_scores[cid[i]] : 0.0f; } while (0)
    for (int i = 0; i < n - 1; i++) PAIR(i);
    while (1) {
        float best_score = -1e10;
        int best_idx = -1;
        for (int i = 0; i < n - 1; i++)
            if (cid[i] != -1 && csc[i] > best_score) { best_score = csc[i]; best_idx = i; }
        if (best_idx == -1) break;
        tokens[best_idx] = cid[best_idx];
        memmove(tokens + best_idx + 1, tokens + best_idx + 2, (size_t)(n - best_idx - 2) * sizeof(int));
        if (n - best_idx - 2 > 0) {
            memmove(cid + best_idx + 1, cid + best_idx + 2, (size_t)(n - best_idx - 2) * sizeof(int));
            memmove(csc + best_idx + 1, csc + best_idx + 2, (size_t)(n - best_idx - 2) * sizeof(float));
        }
        n--;
        if (best_idx < n - 1) PAIR(best_idx);
        if (best_idx > 0) PAIR(best_idx - 1);
    }
#undef PAIR
    *n_tokens = n;
    free(cid); free(csc);
    free(buf);
}

static const char *decode(int prev, int token) {
    const char *piece = vocab[token];
    if (prev == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = (const char *)byte_pieces + byte_val * 2;
    return piece;
}

/* ---------------------------------------------------------- commands */

static void out(const char *s) { fputs(s, stdout); fputc('\n', stdout); fflush(stdout); }

/* parse whitespace-separated ints; returns count or -1 on a bad/out-of-range id */
static int parse_ids(char *s, int **ids) {
    size_t cap = 64; int n = 0;
    int *a = malloc(cap * sizeof(int));
    char *end;
    while (1) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        long v = strtol(s, &end, 10);
        if (end == s || v < 0 || v >= V) { free(a); return -1; }
        if ((size_t)n == cap) { cap *= 2; a = realloc(a, cap * sizeof(int)); }
        a[n++] = (int)v;
        s = end;
    }
    *ids = a;
    return n;
}

/* forward ids at 0..n-1 in chunks; write logits to f and/or record per-position argmax in am */
static void run_fixed(const int *ids, int n, FILE *f, int *am) {
    for (int p0 = 0; p0 < n; p0 += CHUNK) {
        int c = n - p0 < CHUNK ? n - p0 : CHUNK;
        forward_chunk(ids + p0, p0, c);
        for (int r = 0; r < c; r++) {
            if (am) am[p0 + r] = argmax(blogits + (size_t)r * V, V);
        }
        if (f) fwrite(blogits, sizeof(float), (size_t)c * V, f);
    }
}

static void cmd_generate(char *arg) {
    char *end;
    long max_new = strtol(arg, &end, 10);
    if (end == arg) { out("ERR bad GENERATE"); return; }
    const char *text = end;
    if (*text == ' ') text++;
    int *toks = malloc((strlen(text) + 3) * sizeof(int));
    int n;
    encode(text, toks, &n);
    char line[64];
    if (max_new <= 0) { out("DONE 0 max"); free(toks); return; }
    if (n > S) { out("DONE 0 context"); free(toks); return; }
    /* forced prompt tokens (logits computed at every position, as in the reference) */
    int last_r = 0;
    for (int p0 = 0; p0 < n; p0 += CHUNK) {
        int c = n - p0 < CHUNK ? n - p0 : CHUNK;
        forward_chunk(toks + p0, p0, c);
        last_r = c - 1;
    }
    const float *logits = blogits + (size_t)last_r * V;
    long count = 0;
    const char *why;
    while (1) {
        int next = argmax(logits, V);
        if (next == 1) { why = "stop"; break; }
        snprintf(line, sizeof line, "T %d", next);
        out(line);
        count++;
        if (count >= max_new) { why = "max"; break; }
        int pos = n - 1 + (int)count;
        if (pos >= S) { why = "context"; break; }
        forward_chunk(&next, pos, 1);
        logits = blogits;
    }
    snprintf(line, sizeof line, "DONE %ld %s", count, why);
    out(line);
    free(toks);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tokenizer.bin\n", argv[0]); return 1; }
    load_model(argv[1]);
    load_tokenizer(argv[2]);
    out("READY");

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) >= 0) {
        if (len > 0 && line[len - 1] == '\n') line[--len] = 0;
        if (len > 0 && line[len - 1] == '\r') line[--len] = 0;
        if (strcmp(line, "QUIT") == 0) break;
        if (strncmp(line, "ENCODE", 6) == 0 && (line[6] == ' ' || line[6] == 0)) {
            const char *text = line[6] ? line + 7 : line + 6;
            int *toks = malloc((strlen(text) + 3) * sizeof(int));
            int n;
            encode(text, toks, &n);
            fputs("IDS", stdout);
            for (int i = 0; i < n; i++) printf(" %d", toks[i]);
            fputc('\n', stdout);
            fflush(stdout);
            free(toks);
        } else if (strncmp(line, "PIECE ", 6) == 0) {
            int a, b;
            if (sscanf(line + 6, "%d %d", &a, &b) != 2 || a < 0 || a >= V || b < 0 || b >= V) { out("ERR bad PIECE"); continue; }
            const unsigned char *p = (const unsigned char *)decode(a, b);
            fputs("HEX ", stdout);
            for (; *p; p++) printf("%02x", *p);
            fputc('\n', stdout);
            fflush(stdout);
        } else if (strncmp(line, "GENERATE ", 9) == 0) {
            cmd_generate(line + 9);
        } else if (strncmp(line, "FIXED", 5) == 0 && (line[5] == ' ' || line[5] == 0)) {
            int *ids;
            int n = parse_ids(line + 5, &ids);
            if (n < 0 || n > S) { out("ERR bad FIXED"); if (n >= 0) free(ids); continue; }
            int *am = malloc((n + 1) * sizeof(int));
            run_fixed(ids, n, NULL, am);
            printf("DONE %d", n);
            for (int i = 0; i < n; i++) printf(" %d", am[i]);
            fputc('\n', stdout);
            fflush(stdout);
            free(am); free(ids);
        } else if (strncmp(line, "LOGITS ", 7) == 0) {
            char *path = line + 7;
            char *sp = strchr(path, ' ');
            char *rest = "";
            if (sp) { *sp = 0; rest = sp + 1; }
            int *ids;
            int n = parse_ids(rest, &ids);
            if (n < 0 || n > S) { out("ERR bad LOGITS"); if (n >= 0) free(ids); continue; }
            FILE *f = fopen(path, "wb");
            if (!f) { out("ERR cannot open path"); free(ids); continue; }
            run_fixed(ids, n, f, NULL);
            fclose(f);
            printf("DONE %d\n", n);
            fflush(stdout);
            free(ids);
        } else {
            out("ERR unknown command");
        }
    }
    free(line);
#ifdef PROF
    {
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        const char *nm[] = {"other", "qkv", "rope+kvcopy", "attention", "wo", "ffn13", "swiglu", "w2", "cls"};
        double tot = 0, k = (double)tb.numer / tb.denom / 1e6;
        for (int i = 0; i < 9; i++) tot += prof[i];
        for (int i = 0; i < 9; i++) fprintf(stderr, "%-12s %8.1f ms %5.1f%%\n", nm[i], prof[i] * k, 100 * prof[i] / tot);
        fprintf(stderr, "forward calls %ld, total %.1f ms\n", pcalls, tot * k);
    }
#endif
    return 0;
}
