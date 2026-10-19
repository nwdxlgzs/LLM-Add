#include "add.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

static int number(const char *s, int min, int max) {
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    return !errno && *s && !*end && v >= min && v <= max ? (int)v : -1;
}
static int usage(void) {
    fprintf(stderr,
            "ai_add prepare DIR\nai_add train DIR [PRETRAIN=200] [SFT_PER_STAGE=3000] "
            "[RL=400] [BATCH=4]\n"
            "ai_add qat MODEL DIR [STEPS=2000] [BATCH=8]\n"
            "ai_add inspect MODEL\nai_add certify MODEL REPORT [keep|fp32|fp8|fp4] [auto|all]\n"
            "ai_add evaluate MODEL REPORT [RANDOM=10000]\nai_add add MODEL A B\n");
    return 2;
}
int main(int argc, char **argv) {
    if (argc < 3)
        return usage();
    if (!strcmp(argv[1], "prepare"))
        return argc == 3 && add_prepare(argv[2], 20261018) ? 0 : 1;
    if (!strcmp(argv[1], "train")) {
        if (argc > 7)
            return usage();
        add_training c = add_default_training();
        int *fields[] = {&c.pretrain, &c.sft, &c.rl, &c.batch};
        for (int i = 3; i < argc; i++) {
            *fields[i - 3] = number(argv[i], i == 6 ? 1 : 0, i == 6 ? 64 : 1000000);
            if (*fields[i - 3] < 0)
                return usage();
        }
        return add_train(argv[2], c) ? 0 : 1;
    }
    if (!strcmp(argv[1], "qat")) {
        if (argc < 4 || argc > 6)
            return usage();
        int steps = argc >= 5 ? number(argv[4], 1, 1000000) : 2000;
        int batch = argc >= 6 ? number(argv[5], 1, 64) : 8;
        if (steps < 0 || batch < 0)
            return usage();
        return add_qat(argv[2], argv[3], steps, batch) ? 0 : 1;
    }
    char error[256];
    cn_model *m = cn_load(argv[2], error, sizeof(error));
    if (!m) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    int ok = 0;
    if (!strcmp(argv[1], "inspect") && argc == 3) {
        const cn_config *c = cn_model_config(m);
        printf("DSV4.1F reduced model: layers=%d encoder=%d dim=%d parameters=%zu hc=%d "
               "experts=%d+%d vocab=%d precision=%d fp4_weights=%zu fp8_weights=%zu qat=%d\n",
               c->layers, c->encoder_layers, c->dim, cn_parameter_count(m), c->hc, c->experts,
               c->shared_experts, c->vocab, cn_model_precision(m),
               cn_model_quantized_parameters(m, CN_MXFP4), cn_model_quantized_parameters(m, CN_FP8),
               c->qat);
        ok = 1;
    } else if (!strcmp(argv[1], "certify") && argc >= 4 && argc <= 6) {
        cn_dtype dtype = CN_F32;
        if (argc >= 5) {
            if (!strcmp(argv[4], "fp8"))
                dtype = CN_FP8;
            else if (!strcmp(argv[4], "fp4"))
                dtype = CN_MXFP4;
            else if (strcmp(argv[4], "fp32") && strcmp(argv[4], "keep")) {
                cn_model_free(m);
                return usage();
            }
        }
        if (argc == 6 && strcmp(argv[5], "all") && strcmp(argv[5], "auto")) {
            cn_model_free(m);
            return usage();
        }
        ok = (argc < 5 || !strcmp(argv[4], "keep") || cn_model_quantize(m, dtype)) &&
             add_certify(m, argv[2], argv[3], argc == 6 && !strcmp(argv[5], "all"));
        printf("certified=%s\n", ok ? "true" : "false");
    } else if (!strcmp(argv[1], "evaluate") && argc >= 4 && argc <= 5) {
        int count = argc == 5 ? number(argv[4], 0, 10000000) : 10000;
        if (count >= 0)
            ok = add_evaluate(m, argv[3], (unsigned)count, 20261019);
        printf("passed=%s\n", ok ? "true" : "false");
    } else if (!strcmp(argv[1], "add") && argc == 5) {
        add_result r = add_check(m, NULL);
        if (r.correct != 200)
            fprintf(stderr, "refused: only %d/200 local states certified\n", r.correct);
        else {
            cn_context *s = cn_context_create(m, 0);
            add_trace t = {0};
            if (s && add_predict(s, argv[3], argv[4], &t, NULL))
                ok = add_print(stdout, argv[3], argv[4], &t);
            cn_context_free(s);
        }
    } else {
        cn_model_free(m);
        return usage();
    }
    cn_model_free(m);
    return ok ? 0 : 1;
}
