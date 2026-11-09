/* Test driver around the supplied reference (reference/run.c, compiled with -DTESTING; not modified).
 *   refdrv model.bin tokenizer.bin logits <out.f32> <id> <id> ...   reference forward() logits per position
 *   refdrv model.bin tokenizer.bin encode                            stdin lines -> "IDS ..." via reference encode()
 */
#define TESTING
#include "../reference/run.c"
int main(int argc, char **argv) {
    if (argc < 4) return 1;
    Transformer t; build_transformer(&t, argv[1]);
    Tokenizer tk; build_tokenizer(&tk, argv[2], t.config.vocab_size);
    if (!strcmp(argv[3], "logits")) {
        FILE *f = fopen(argv[4], "wb");
        for (int i = 5; i < argc; i++) fwrite(forward(&t, atoi(argv[i]), i - 5), sizeof(float), t.config.vocab_size, f);
        fclose(f);
    } else {
        static char line[1 << 16]; int toks[1 << 16], n;
        while (fgets(line, sizeof line, stdin)) {
            size_t L = strlen(line); if (L && line[L - 1] == '\n') line[--L] = 0;
            encode(&tk, line, 1, 0, toks, &n);
            printf("IDS"); for (int i = 0; i < n; i++) printf(" %d", toks[i]); printf("\n");
        }
    }
    return 0;
}
