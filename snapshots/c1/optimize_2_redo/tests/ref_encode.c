/* differential helper: links the unmodified reference (compiled with -DTESTING)
   and prints encode()/decode() results for lines on stdin. */
#define TESTING
#include "../reference/run.c"
int main(int argc, char **argv) {
    Tokenizer t; build_tokenizer(&t, argv[1], 32000);
    char *line = NULL; size_t cap = 0; ssize_t len;
    int *toks = malloc(1 << 20);
    while ((len = getline(&line, &cap, stdin)) >= 0) {
        if (len && line[len-1] == '\n') line[--len] = 0;
        int n; encode(&t, line, 1, 0, toks, &n);
        printf("IDS"); for (int i = 0; i < n; i++) printf(" %d", toks[i]); printf("\n");
    }
    for (int prev = 0; prev < 3; prev++) for (int id = 0; id < 32000; id++) {
        const unsigned char *p = (const unsigned char *)decode(&t, prev, id);
        printf("HEX "); for (; *p; p++) printf("%02x", *p); printf("\n");
    }
    return 0;
}
