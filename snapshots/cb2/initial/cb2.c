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

/* ---------------------------------------------------------------- model */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

static Config cfg;
static int dim, hid, nl, nh, nkv, V, S, hs, kvd, qkvd, kv_mul;
static float *emb, *wcls, *rms_att, *rms_ffn, *rms_final;
static float **wqkv, **wo, **w13, **w2; /* per-layer pointers into one block */
static float *kcache, *vcache;          /* (layer, seq_len, kv_dim) */
static float *rope_cos, *rope_sin;      /* (seq_len, head_size/2) */

#define CHUNK 512 /* max positions forwarded together */
static float *bx, *bxb, *bxb2, *bqkv, *bh13, *bhb, *batt, *blogits;

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
    int shared = cfg.vocab_size > 0;
    cfg.vocab_size = abs(cfg.vocab_size);
    dim = cfg.dim; hid = cfg.hidden_dim; nl = cfg.n_layers; nh = cfg.n_heads; nkv = cfg.n_kv_heads;
    V = cfg.vocab_size; S = cfg.seq_len;
    hs = dim / nh; kvd = dim * nkv / nh; kv_mul = nh / nkv; qkvd = dim + 2 * kvd;

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

    emb = xalloc(VV * D * 4);
    pread_all(fd, emb, VV * D * 4, OFF(o_emb));
    rms_att = xalloc(L * D * 4);
    pread_all(fd, rms_att, L * D * 4, OFF(o_rmsatt));
    rms_ffn = xalloc(L * D * 4);
    pread_all(fd, rms_ffn, L * D * 4, OFF(o_rmsffn));
    rms_final = xalloc(D * 4);
    pread_all(fd, rms_final, D * 4, OFF(o_rmsfin));
    if (shared) wcls = emb;
    else { wcls = xalloc(VV * D * 4); pread_all(fd, wcls, VV * D * 4, OFF(o_cls)); }

    /* per-layer block: [wq; wk; wv] (qkvd x dim), wo (dim x dim), [w1; w3] (2*hid x dim), w2 (dim x hid) */
    size_t per = (size_t)qkvd * D + D * D + 2 * H * D + D * H;
    float *blk = xalloc(L * per * 4);
    wqkv = malloc(L * sizeof(float *)); wo = malloc(L * sizeof(float *));
    w13 = malloc(L * sizeof(float *)); w2 = malloc(L * sizeof(float *));
    for (size_t l = 0; l < L; l++) {
        float *b = blk + l * per;
        wqkv[l] = b; b += (size_t)qkvd * D;
        wo[l] = b; b += D * D;
        w13[l] = b; b += 2 * H * D;
        w2[l] = b;
        pread_all(fd, wqkv[l], D * D * 4, OFF(o_wq + l * D * D));
        pread_all(fd, wqkv[l] + D * D, KV * D * 4, OFF(o_wk + l * KV * D));
        pread_all(fd, wqkv[l] + D * D + KV * D, KV * D * 4, OFF(o_wv + l * KV * D));
        pread_all(fd, wo[l], D * D * 4, OFF(o_wo + l * D * D));
        pread_all(fd, w13[l], H * D * 4, OFF(o_w1 + l * H * D));
        pread_all(fd, w13[l] + H * D, H * D * 4, OFF(o_w3 + l * H * D));
        pread_all(fd, w2[l], D * H * 4, OFF(o_w2 + l * D * H));
    }
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
    bqkv = xalloc(C * qkvd * 4); bh13 = xalloc(C * 2 * H * 4); bhb = xalloc(C * H * 4);
    batt = xalloc(C * (size_t)S * 4); blogits = xalloc(C * VV * 4);
}

/* ------------------------------------------------------------ forward */

/* y (c x d) = x (c x n) @ W^T, W is (d x n) row-major */
static void matmul(float *y, const float *x, const float *W, int c, int n, int d) {
    if (c == 1)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, W, n, x, 1, 0.0f, y, 1);
    else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, d, n, 1.0f, x, n, W, n, 0.0f, y, d);
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
    for (int r = 0; r < c; r++) memcpy(bx + (size_t)r * dim, emb + (size_t)toks[r] * dim, dim * 4);

    for (int l = 0; l < nl; l++) {
        float *kc = kcache + (size_t)l * S * kvd, *vc = vcache + (size_t)l * S * kvd;
        for (int r = 0; r < c; r++) rmsnorm(bxb + (size_t)r * dim, bx + (size_t)r * dim, rms_att + (size_t)l * dim, dim);
        matmul(bqkv, bxb, wqkv[l], c, dim, qkvd);
        for (int r = 0; r < c; r++) {
            float *q = bqkv + (size_t)r * qkvd, *k = q + dim, *v = k + kvd;
            int pos = p0 + r;
            rope(q, dim, pos);
            rope(k, kvd, pos);
            memcpy(kc + (size_t)pos * kvd, k, kvd * 4);
            memcpy(vc + (size_t)pos * kvd, v, kvd * 4);
        }
        for (int h = 0; h < nh; h++) {
            const float *qh = bqkv + h * hs;
            const float *kh = kc + (h / kv_mul) * hs, *vh = vc + (h / kv_mul) * hs;
            float *oh = bxb + h * hs;
            if (c == 1) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, T, hs, 1.0f, kh, kvd, qh, 1, 0.0f, batt, 1);
                vDSP_vsdiv(batt, 1, &sq, batt, 1, T);
                softmax(batt, T);
                cblas_sgemv(CblasRowMajor, CblasTrans, T, hs, 1.0f, vh, kvd, batt, 1, 0.0f, oh, 1);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, T, hs, 1.0f, qh, qkvd, kh, kvd,
                            0.0f, batt, T);
                for (int r = 0; r < c; r++) {
                    float *row = batt + (size_t)r * T;
                    int n = p0 + r + 1; /* causal: keys 0..pos */
                    vDSP_vsdiv(row, 1, &sq, row, 1, n);
                    softmax(row, n);
                    if (n < T) memset(row + n, 0, (size_t)(T - n) * 4);
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, hs, T, 1.0f, batt, T, vh, kvd,
                            0.0f, oh, dim);
            }
        }
        matmul(bxb2, bxb, wo[l], c, dim, dim);
        vDSP_vadd(bx, 1, bxb2, 1, bx, 1, (vDSP_Length)c * dim);

        for (int r = 0; r < c; r++) rmsnorm(bxb + (size_t)r * dim, bx + (size_t)r * dim, rms_ffn + (size_t)l * dim, dim);
        matmul(bh13, bxb, w13[l], c, dim, 2 * hid);
        for (int r = 0; r < c; r++) {
            const float *a = bh13 + (size_t)r * 2 * hid, *b = a + hid;
            float *o = bhb + (size_t)r * hid;
            for (int i = 0; i < hid; i++) {
                float val = a[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= b[i];
                o[i] = val;
            }
        }
        matmul(bxb, bhb, w2[l], c, hid, dim);
        vDSP_vadd(bx, 1, bxb, 1, bx, 1, (vDSP_Length)c * dim);
    }
    for (int r = 0; r < c; r++) rmsnorm(bx + (size_t)r * dim, bx + (size_t)r * dim, rms_final, dim);
    matmul(blogits, bx, wcls, c, dim, V);
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
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            sprintf(buf, "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
            int id = str_lookup(buf);
            if (id != -1 && vocab_scores[id] > best_score) { best_score = vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
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
    return 0;
}
