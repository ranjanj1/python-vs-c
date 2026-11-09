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
#include <sys/mman.h>
#include <unistd.h>

/* ------------------------------------------------------------------ model */

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    float *token_embedding_table, *rms_att_weight, *rms_ffn_weight;
    float *wq, *wk, *wv, *wo, *w1, *w2, *w3, *rms_final_weight, *wcls;
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

static void load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(&C, sizeof(Config), 1, f) != 1) { fprintf(stderr, "bad header\n"); exit(1); }
    int shared = C.vocab_size > 0;
    C.vocab_size = abs(C.vocab_size);
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fclose(f);
    int fd = open(path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed\n"); exit(1); }
    float *data = mmap(NULL, (size_t)fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(1); }

    head_size = C.dim / C.n_heads;
    kv_dim = (C.dim * C.n_kv_heads) / C.n_heads;
    kv_mul = C.n_heads / C.n_kv_heads;
    unsigned long long L = (unsigned long long)C.n_layers;
    /* Copy the float32 weights (unchanged) into a page-aligned buffer: the file
     * places them at byte offset 28, and aligned rows measured ~10% faster in sgemv. */
    size_t nfl = ((size_t)fsize - sizeof(Config)) / sizeof(float);
    float *p = NULL;
    if (posix_memalign((void **)&p, 16384, nfl * sizeof(float))) { fprintf(stderr, "malloc failed\n"); exit(1); }
    memcpy(p, data + sizeof(Config) / sizeof(float), nfl * sizeof(float));
    munmap(data, (size_t)fsize);
    close(fd);
    W.token_embedding_table = p; p += (size_t)C.vocab_size * C.dim;
    W.rms_att_weight = p;        p += L * C.dim;
    W.wq = p;                    p += L * C.dim * (C.n_heads * head_size);
    W.wk = p;                    p += L * C.dim * (C.n_kv_heads * head_size);
    W.wv = p;                    p += L * C.dim * (C.n_kv_heads * head_size);
    W.wo = p;                    p += L * (C.n_heads * head_size) * C.dim;
    W.rms_ffn_weight = p;        p += L * C.dim;
    W.w1 = p;                    p += L * C.dim * C.hidden_dim;
    W.w2 = p;                    p += L * C.hidden_dim * C.dim;
    W.w3 = p;                    p += L * C.dim * C.hidden_dim;
    W.rms_final_weight = p;      p += C.dim;
    p += C.seq_len * head_size / 2; /* legacy freq_cis_real */
    p += C.seq_len * head_size / 2; /* legacy freq_cis_imag */
    W.wcls = shared ? W.token_embedding_table : p;

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

    for (unsigned long long l = 0; l < (unsigned long long)C.n_layers; l++) {
        rmsnorm(S.xb, x, W.rms_att_weight + l * dim, dim);

        size_t loff = l * (size_t)C.seq_len * kv_dim;
        S.k = S.key_cache + loff + (size_t)pos * kv_dim;
        S.v = S.value_cache + loff + (size_t)pos * kv_dim;

        matmul(S.q, S.xb, W.wq + l * dim * dim, dim, dim);
        matmul(S.k, S.xb, W.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(S.v, S.xb, W.wv + l * dim * kv_dim, dim, kv_dim);

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

        float scale = sqrtf((float)head_size);
        for (int h = 0; h < C.n_heads; h++) {
            const float *q = S.q + h * head_size;
            float *att = S.att + (size_t)h * C.seq_len;
            const float *kbase = S.key_cache + loff + (h / kv_mul) * head_size;
            const float *vbase = S.value_cache + loff + (h / kv_mul) * head_size;
            /* scores: K[0..pos] (pos+1, head_size; stride kv_dim) @ q */
            cblas_sgemv(CblasRowMajor, CblasNoTrans, pos + 1, head_size, 1.0f,
                        kbase, kv_dim, q, 1, 0.0f, att, 1);
            for (int t = 0; t <= pos; t++) att[t] /= scale;
            softmax(att, pos + 1);
            /* weighted value sum: V[0..pos]^T @ att -> xb (head slice) */
            cblas_sgemv(CblasRowMajor, CblasTrans, pos + 1, head_size, 1.0f,
                        vbase, kv_dim, att, 1, 0.0f, S.xb + h * head_size, 1);
        }

        matmul(S.xb2, S.xb, W.wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += S.xb2[i];

        rmsnorm(S.xb, x, W.rms_ffn_weight + l * dim, dim);

        matmul(S.hb, S.xb, W.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(S.hb2, S.xb, W.w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = S.hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= S.hb2[i];
            S.hb[i] = val;
        }
        matmul(S.xb, S.hb, W.w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += S.xb[i];
    }

    rmsnorm(x, x, W.rms_final_weight, dim);
    matmul(S.logits, x, W.wcls, C.dim, C.vocab_size);
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
    return 0;
}
