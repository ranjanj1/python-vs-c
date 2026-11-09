// C0: the pinned upstream run.c, unchanged, compiled in through its own TESTING switch. Only the command loop below is added.
#define TESTING
#include "../../sources/upstream/run.c"

static int argmax(float *x, int n) { int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b; }

static void reset_state(Transformer *t) {
    Config *p = &t->config;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    size_t n = (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float);
    memset(t->state.key_cache, 0, n); memset(t->state.value_cache, 0, n);
}

static int parse_ids(char *s, int *ids, int cap) {
    int n = 0; char *tok = strtok(s, " \n");
    while (tok && n < cap) { ids[n++] = atoi(tok); tok = strtok(NULL, " \n"); }
    return n;
}

int main(int argc, char **argv) {
    Transformer t; Tokenizer k;
    build_transformer(&t, argv[1]);
    build_tokenizer(&k, argv[2], t.config.vocab_size);
    int V = t.config.vocab_size, S = t.config.seq_len;
    size_t cap = 1 << 22; char *line = malloc(cap);
    int *ids = malloc(sizeof(int) * (1 << 20));
    puts("READY"); fflush(stdout);
    while (fgets(line, cap, stdin)) {
        line[strcspn(line, "\n")] = 0;
        if (!strncmp(line, "ENCODE ", 7)) {
            int n = 0; encode(&k, line + 7, 1, 0, ids, &n);
            printf("IDS"); for (int i = 0; i < n; i++) printf(" %d", ids[i]); puts("");
        } else if (!strncmp(line, "PIECE ", 6)) {
            int a, b; sscanf(line + 6, "%d %d", &a, &b);
            char *p = decode(&k, a, b);
            printf("HEX "); for (; *p; p++) printf("%02x", (unsigned char)*p); puts("");
        } else if (!strncmp(line, "GENERATE ", 9)) {
            char *sp = strchr(line + 9, ' '); int max_new = atoi(line + 9); char *text = sp + 1;
            int n = 0; encode(&k, text, 1, 0, ids, &n); reset_state(&t);
            int token = ids[0], pos = 0, made = 0; const char *why = "context";
            while (pos < S) {
                float *logits = forward(&t, token, pos);
                if (pos < n - 1) { token = ids[++pos]; continue; }
                int next = argmax(logits, V); pos++;
                if (next == 1) { why = "stop"; break; }
                printf("T %d\n", next); fflush(stdout); made++; token = next;
                if (made >= max_new) { why = "max"; break; }
            }
            printf("DONE %d %s\n", made, why);
        } else if (!strncmp(line, "FIXED ", 6)) {
            int n = parse_ids(line + 6, ids, 1 << 20); reset_state(&t);
            int *am = malloc(sizeof(int) * n);
            for (int i = 0; i < n; i++) am[i] = argmax(forward(&t, ids[i], i), V);
            printf("DONE %d", n); for (int i = 0; i < n; i++) printf(" %d", am[i]); puts(""); free(am);
        } else if (!strncmp(line, "LOGITS ", 7)) {
            char *sp = strchr(line + 7, ' '); *sp = 0; FILE *f = fopen(line + 7, "wb");
            int n = parse_ids(sp + 1, ids, 1 << 20); reset_state(&t);
            for (int i = 0; i < n; i++) fwrite(forward(&t, ids[i], i), sizeof(float), V, f);
            fclose(f); printf("DONE %d\n", n);
        } else if (!strcmp(line, "QUIT")) { return 0; }
        fflush(stdout);
    }
    return 0;
}
