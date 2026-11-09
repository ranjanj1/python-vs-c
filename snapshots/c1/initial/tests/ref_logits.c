/* differential helper: unmodified reference forward(); writes n x vocab logits.
   usage: ref_logits model.bin out.f32 id id ... */
#define TESTING
#include "../reference/run.c"
int main(int argc, char **argv) {
    Transformer t; build_transformer(&t, argv[1]);
    FILE *f = fopen(argv[2], "wb");
    for (int i = 3; i < argc; i++)
        fwrite(forward(&t, atoi(argv[i]), i - 3), 4, t.config.vocab_size, f);
    fclose(f); return 0;
}
