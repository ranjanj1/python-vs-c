/* CB port of reference/run.c (llama2.c): C + Apple Accelerate BLAS, float32.
 * Matrix-vector products use cblas_sgemv with the same (d,n) @ (n,) shapes as the
 * reference matmul; everything else follows the reference loops. Single sequence,
 * batch size 1, greedy decoding, stdin/stdout protocol from INTERFACE.md. */
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Optional section profiler: build with -DCB_PROFILE; totals go to stderr on QUIT. */
#ifdef CB_PROFILE
#include <time.h>
enum { P_EMB, P_RMS, P_QKV, P_ROPE, P_ATT, P_WO, P_W13, P_SWI, P_W2, P_RES, P_CLS, P_N };
static const char *PNAME[P_N] = {"embed+rope-angles", "rmsnorm", "wq/wk/wv sgemv", "rope", "attention",
                                 "wo sgemv", "w1/w3 sgemv", "swiglu", "w2 sgemv", "residual", "classifier sgemv"};
static double PT[P_N];
static long PFWD;
#define PSTART uint64_t _pt = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#define PMARK(i) do { uint64_t _pu = clock_gettime_nsec_np(CLOCK_UPTIME_RAW); PT[i] += (_pu - _pt) * 1e-9; _pt = _pu; } while (0)
static void pdump(void) {
    double tot = 0;
    for (int i = 0; i < P_N; i++) tot += PT[i];
    fprintf(stderr, "profile: %ld forwards, %.3f ms/forward\n", PFWD, tot / (PFWD ? PFWD : 1) * 1e3);
    for (int i = 0; i < P_N; i++)
        fprintf(stderr, "  %-18s %8.3f s %5.1f%% %7.3f ms/fwd\n", PNAME[i], PT[i], 100 * PT[i] / tot, PT[i] / (PFWD ? PFWD : 1) * 1e3);
}
#else
#define PSTART
#define PMARK(i) do { } while (0)
#define pdump() do { } while (0)
#endif

/* ------------------------------------------------------------------ model */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float *rms_att, *rms_ffn, *wq, *wk, *wv, *wo, *w1, *w3, *w2;
} Layer;

typedef struct {
    float *token_embedding_table, *rms_final_weight, *wcls;
    Layer *layer;
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
    float *rope_cos, *rope_sin; /* (dim/2,) per-forward scratch */
} RunState;

static Config C;
static Weights W;
static RunState S;
static int kv_dim, kv_mul, head_size;
static int cache_used = 0; /* positions written since the last cache reset */

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "malloc failed\n"); exit(1); }
    return p;
}

static void read_floats(int fd, float *dst, size_t off_floats, size_t n) {
    char *d = (char *)dst;
    size_t left = n * sizeof(float);
    off_t off = (off_t)(off_floats * sizeof(float));
    while (left > 0) {
        size_t chunk = left > ((size_t)1 << 30) ? ((size_t)1 << 30) : left;
        ssize_t r = pread(fd, d, chunk, off);
        if (r <= 0) { fprintf(stderr, "short read on checkpoint\n"); exit(1); }
        d += r; off += r; left -= (size_t)r;
    }
}

static void load_model(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (pread(fd, &C, sizeof(Config), 0) != (ssize_t)sizeof(Config)) { fprintf(stderr, "bad header\n"); exit(1); }
    int shared = C.vocab_size > 0;
    C.vocab_size = abs(C.vocab_size);

    head_size = C.dim / C.n_heads;
    kv_dim = (C.dim * C.n_kv_heads) / C.n_heads;
    kv_mul = C.n_heads / C.n_kv_heads;
    size_t L = (size_t)C.n_layers, dim = C.dim, hd = C.hidden_dim, V = C.vocab_size;
    size_t qd = (size_t)C.n_heads * head_size, kvd = (size_t)C.n_kv_heads * head_size;

    /* Float offsets of each tensor in the checkpoint (same layout as the reference's
     * memory_map_weights). */
    size_t o = sizeof(Config) / sizeof(float);
    size_t o_emb = o;   o += V * dim;
    size_t o_ratt = o;  o += L * dim;
    size_t o_wq = o;    o += L * dim * qd;
    size_t o_wk = o;    o += L * dim * kvd;
    size_t o_wv = o;    o += L * dim * kvd;
    size_t o_wo = o;    o += L * qd * dim;
    size_t o_rffn = o;  o += L * dim;
    size_t o_w1 = o;    o += L * dim * hd;
    size_t o_w2 = o;    o += L * hd * dim;
    size_t o_w3 = o;    o += L * dim * hd;
    size_t o_rfin = o;  o += dim;
    o += (size_t)C.seq_len * head_size / 2; /* legacy freq_cis_real */
    o += (size_t)C.seq_len * head_size / 2; /* legacy freq_cis_imag */
    size_t o_cls = o;

    /* The same float32 values are read into one page-aligned buffer, re-laid-out in
     * the order forward() streams them: per layer wq,wk,wv,wo,w1,w3,w2, then the
     * classifier. (Measured: the file is at a 28-byte offset and splits each tensor
     * type across all layers; aligned, access-ordered weights sgemv faster.) */
    size_t per_layer = 2 * dim + dim * qd + 2 * dim * kvd + qd * dim + 3 * dim * hd;
    size_t total = V * dim + dim + L * per_layer + (shared ? 0 : V * dim);
    float *buf = NULL;
    if (posix_memalign((void **)&buf, 16384, total * sizeof(float))) { fprintf(stderr, "malloc failed\n"); exit(1); }
    float *p = buf;
#define TAKE(dst, off, n) do { (dst) = p; read_floats(fd, p, (off), (n)); p += (n); } while (0)
    W.layer = xcalloc(L, sizeof(Layer));
    TAKE(W.token_embedding_table, o_emb, V * dim);
    for (size_t l = 0; l < L; l++) {
        Layer *ly = &W.layer[l];
        TAKE(ly->rms_att, o_ratt + l * dim, dim);
        TAKE(ly->rms_ffn, o_rffn + l * dim, dim);
        TAKE(ly->wq, o_wq + l * dim * qd, dim * qd);
        TAKE(ly->wk, o_wk + l * dim * kvd, dim * kvd);
        TAKE(ly->wv, o_wv + l * dim * kvd, dim * kvd);
        TAKE(ly->wo, o_wo + l * qd * dim, qd * dim);
        TAKE(ly->w1, o_w1 + l * dim * hd, dim * hd);
        TAKE(ly->w3, o_w3 + l * dim * hd, dim * hd);
        TAKE(ly->w2, o_w2 + l * hd * dim, hd * dim);
    }
    TAKE(W.rms_final_weight, o_rfin, dim);
    if (shared) W.wcls = W.token_embedding_table;
    else TAKE(W.wcls, o_cls, V * dim);
#undef TAKE
    close(fd);

    S.x = xcalloc(C.dim, sizeof(float));
    S.xb = xcalloc(C.dim, sizeof(float));
    S.xb2 = xcalloc(C.dim, sizeof(float));
    S.hb = xcalloc(C.hidden_dim, sizeof(float));
    S.hb2 = xcalloc(C.hidden_dim, sizeof(float));
    S.q = xcalloc(C.dim, sizeof(float));
    S.key_cache = xcalloc((size_t)C.n_layers * C.seq_len * kv_dim, sizeof(float));
    S.value_cache = xcalloc((size_t)C.n_layers * C.seq_len * kv_dim, sizeof(float));
    S.att = xcalloc((size_t)C.n_heads * C.seq_len, sizeof(float));
    S.logits = xcalloc(C.vocab_size, sizeof(float));
    S.rope_cos = xcalloc(C.dim / 2, sizeof(float));
    S.rope_sin = xcalloc(C.dim / 2, sizeof(float));
}

/* Every command starts from a clean KV cache: zero the rows the previous command wrote. */
static void reset_cache(void) {
    if (cache_used > 0) {
        for (int l = 0; l < C.n_layers; l++) {
            size_t off = (size_t)l * C.seq_len * kv_dim;
            memset(S.key_cache + off, 0, (size_t)cache_used * kv_dim * sizeof(float));
            memset(S.value_cache + off, 0, (size_t)cache_used * kv_dim * sizeof(float));
        }
    }
    cache_used = 0;
}

/* ------------------------------------------------------------ nn blocks */

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

/* W (d,n) @ x (n,) -> xout (d,), row-major, via Accelerate BLAS */
static void matmul(float *xout, const float *x, const float *w, int n, int d) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d, n, 1.0f, w, n, x, 1, 0.0f, xout, 1);
}

static float *forward(int token, int pos) {
    float *x = S.x;
    int dim = C.dim, hidden_dim = C.hidden_dim;

    PSTART
    memcpy(x, W.token_embedding_table + (size_t)token * dim, dim * sizeof(float));

    /* RoPE angles: same per-element formula as the reference; values do not depend
     * on the layer, so they are evaluated once per forward instead of once per layer. */
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        S.rope_cos[i / 2] = cosf(val);
        S.rope_sin[i / 2] = sinf(val);
    }
    PMARK(P_EMB);

    for (unsigned long long l = 0; l < (unsigned long long)C.n_layers; l++) {
        const Layer *ly = &W.layer[l];
        rmsnorm(S.xb, x, ly->rms_att, dim);
        PMARK(P_RMS);

        /* key and value point into the kv cache, (layer, seq_len, kv_dim) as in the reference */
        size_t loff = l * (size_t)C.seq_len * kv_dim;
        S.k = S.key_cache + loff + (size_t)pos * kv_dim;
        S.v = S.value_cache + loff + (size_t)pos * kv_dim;

        matmul(S.q, S.xb, ly->wq, dim, dim);
        matmul(S.k, S.xb, ly->wk, dim, kv_dim);
        matmul(S.v, S.xb, ly->wv, dim, kv_dim);
        PMARK(P_QKV);

        for (int i = 0; i < dim; i += 2) {
            float fcr = S.rope_cos[i / 2], fci = S.rope_sin[i / 2];
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? S.q : S.k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        PMARK(P_ROPE);
        float scale = sqrtf((float)head_size);
        for (int h = 0; h < C.n_heads; h++) {
            const float *q = S.q + h * head_size;
            float *att = S.att + (size_t)h * C.seq_len;
            const float *kbase = S.key_cache + loff + (h / kv_mul) * head_size;
            const float *vbase = S.value_cache + loff + (h / kv_mul) * head_size;
            /* scores: K[0..pos] (pos+1, head_size; row stride kv_dim) @ q */
            cblas_sgemv(CblasRowMajor, CblasNoTrans, pos + 1, head_size, 1.0f,
                        kbase, kv_dim, q, 1, 0.0f, att, 1);
            for (int t = 0; t <= pos; t++) att[t] /= scale;
            softmax(att, pos + 1);
            /* weighted value sum: V[0..pos]^T @ att -> xb (head slice) */
            cblas_sgemv(CblasRowMajor, CblasTrans, pos + 1, head_size, 1.0f,
                        vbase, kv_dim, att, 1, 0.0f, S.xb + h * head_size, 1);
        }

        PMARK(P_ATT);
        matmul(S.xb2, S.xb, ly->wo, dim, dim);
        PMARK(P_WO);
        for (int i = 0; i < dim; i++) x[i] += S.xb2[i];
        PMARK(P_RES);

        rmsnorm(S.xb, x, ly->rms_ffn, dim);
        PMARK(P_RMS);

        matmul(S.hb, S.xb, ly->w1, dim, hidden_dim);
        matmul(S.hb2, S.xb, ly->w3, dim, hidden_dim);
        PMARK(P_W13);
        for (int i = 0; i < hidden_dim; i++) {
            float val = S.hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= S.hb2[i];
            S.hb[i] = val;
        }
        PMARK(P_SWI);
        matmul(S.xb, S.hb, ly->w2, hidden_dim, dim);
        PMARK(P_W2);
        for (int i = 0; i < dim; i++) x[i] += S.xb[i];
        PMARK(P_RES);
    }

    rmsnorm(x, x, W.rms_final_weight, dim);
    PMARK(P_RMS);
    matmul(S.logits, x, W.wcls, C.dim, C.vocab_size);
    PMARK(P_CLS);
#ifdef CB_PROFILE
    PFWD++;
#endif
    if (pos + 1 > cache_used) cache_used = pos + 1;
    return S.logits;
}

static int argmax(const float *v, int n) {
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_p) { max_i = i; max_p = v[i]; }
    return max_i;
}

/* -------------------------------------------------------------- tokenizer */

typedef struct { char *str; int id; } TokenIndex;

static char **vocab;
static float *vocab_scores;
static TokenIndex *sorted_vocab;
static unsigned int max_token_length;
static unsigned char byte_pieces[512];

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

static void load_tokenizer(const char *path, int vocab_size) {
    vocab = malloc(vocab_size * sizeof(char *));
    vocab_scores = malloc(vocab_size * sizeof(float));
    for (int i = 0; i < 256; i++) { byte_pieces[i * 2] = (unsigned char)i; byte_pieces[i * 2 + 1] = '\0'; }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "couldn't load %s\n", path); exit(1); }
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(vocab_scores + i, sizeof(float), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        if (fread(&len, sizeof(int), 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        vocab[i] = malloc(len + 1);
        if (fread(vocab[i], len, 1, f) != 1) { fprintf(stderr, "failed read\n"); exit(1); }
        vocab[i][len] = '\0';
    }
    fclose(f);
    /* the reference builds this lazily on first encode; doing it at load is equivalent */
    sorted_vocab = malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) { sorted_vocab[i].str = vocab[i]; sorted_vocab[i].id = i; }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
}

static const char *decode(int prev_token, int token) {
    const char *piece = vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = (const char *)byte_pieces + byte_val * 2;
    return piece;
}

static int str_lookup(char *str) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, C.vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

/* Same algorithm as reference encode() with bos=1, eos=0. */
static void encode(const char *text, int *tokens, int *n_tokens) {
    char *str_buffer = malloc((max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ");
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id]; best_id = id; best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    free(str_buffer);
}

/* --------------------------------------------------------------- protocol */

static int parse_ids(char *s, int **out) {
    int cap = 64, n = 0;
    int *ids = malloc(cap * sizeof(int));
    char *end;
    while (*s) {
        while (*s == ' ') s++;
        if (!*s) break;
        long v = strtol(s, &end, 10);
        if (end == s) break;
        if (n == cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
        ids[n++] = (int)v;
        s = end;
    }
    *out = ids;
    return n;
}

static int valid_id(int id) { return id >= 0 && id < C.vocab_size; }

static void cmd_encode(const char *text) {
    int *tokens = malloc((strlen(text) + 3) * sizeof(int));
    int n;
    encode(text, tokens, &n);
    fputs("IDS", stdout);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    putchar('\n');
    fflush(stdout);
    free(tokens);
}

static void cmd_piece(char *args) {
    int *ids;
    int n = parse_ids(args, &ids);
    if (n < 2 || !valid_id(ids[0]) || !valid_id(ids[1])) { puts("ERR bad PIECE"); fflush(stdout); free(ids); return; }
    const unsigned char *p = (const unsigned char *)decode(ids[0], ids[1]);
    fputs("HEX ", stdout);
    for (; *p; p++) printf("%02x", *p);
    putchar('\n');
    fflush(stdout);
    free(ids);
}

static void cmd_generate(char *args) {
    char *end;
    long max_new = strtol(args, &end, 10);
    const char *text = (*end == ' ') ? end + 1 : end;
    int *tokens = malloc((strlen(text) + 3) * sizeof(int));
    int n;
    encode(text, tokens, &n);
    reset_cache();

    int count = 0;
    const char *why = "max";
    if (max_new > 0) {
        int token = tokens[0], pos = 0;
        why = "context";
        while (pos < C.seq_len) {
            float *logits = forward(token, pos);
            int next;
            int forced = pos < n - 1;
            next = forced ? tokens[pos + 1] : argmax(logits, C.vocab_size);
            pos++;
            if (next == 1) { why = "stop"; break; }
            if (!forced) {
                printf("T %d\n", next);
                fflush(stdout);
                count++;
                if (count >= max_new) { why = "max"; break; }
            }
            token = next;
        }
    }
    printf("DONE %d %s\n", count, why);
    fflush(stdout);
    free(tokens);
}

/* FIXED / LOGITS: forward ids at positions 0..n-1 (at most seq_len). */
static void cmd_fixed(char *args, const char *path) {
    int *ids;
    int n = parse_ids(args, &ids);
    if (n > C.seq_len) n = C.seq_len;
    for (int i = 0; i < n; i++)
        if (!valid_id(ids[i])) { puts("ERR bad id"); fflush(stdout); free(ids); return; }
    FILE *f = NULL;
    if (path) {
        f = fopen(path, "wb");
        if (!f) { puts("ERR cannot open path"); fflush(stdout); free(ids); return; }
    }
    int *am = malloc((n > 0 ? n : 1) * sizeof(int));
    reset_cache();
    for (int pos = 0; pos < n; pos++) {
        float *logits = forward(ids[pos], pos);
        if (f) fwrite(logits, sizeof(float), C.vocab_size, f);
        else am[pos] = argmax(logits, C.vocab_size);
    }
    if (f) {
        fclose(f);
        printf("DONE %d\n", n);
    } else {
        printf("DONE %d", n);
        for (int i = 0; i < n; i++) printf(" %d", am[i]);
        putchar('\n');
    }
    fflush(stdout);
    free(am);
    free(ids);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tokenizer.bin\n", argv[0]); return 1; }
    load_model(argv[1]);
    load_tokenizer(argv[2], C.vocab_size);
    puts("READY");
    fflush(stdout);

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) != -1) {
        if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
        char *sp = strchr(line, ' ');
        size_t clen = sp ? (size_t)(sp - line) : (size_t)len;
        char *rest = sp ? sp + 1 : line + len;
#define IS(name) (clen == strlen(name) && strncmp(line, name, clen) == 0)
        if (IS("ENCODE")) cmd_encode(rest);
        else if (IS("PIECE")) cmd_piece(rest);
        else if (IS("GENERATE")) cmd_generate(rest);
        else if (IS("FIXED")) cmd_fixed(rest, NULL);
        else if (IS("LOGITS")) {
            char *sp2 = strchr(rest, ' ');
            if (sp2) *sp2 = '\0';
            cmd_fixed(sp2 ? sp2 + 1 : rest + strlen(rest), rest);
        }
        else if (IS("QUIT")) break;
        else if (len == 0) continue;
        else { puts("ERR unknown command"); fflush(stdout); }
#undef IS
    }
    free(line);
    pdump();
    return 0;
}
