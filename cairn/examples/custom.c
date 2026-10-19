#include "cairn.h"
#include <stdio.h>
int main(void) {
    /* Start from a valid, fully specified configuration, then change its structure. */
    cn_config config;
    cn_config_tiny(&config);
    config.dim = 64;
    config.ff_dim = 96;
    config.hc = 4;
    config.layer[3].index_source = 2; /* Reuse: consume layer 2's selection */
    config.layer[1].engram = (cn_engram_config){
        .enabled = 1, .max_order = 4, .heads = 2, .head_dim = 8, .buckets = 127, .pad_id = CN_EOT};
    char error[256];
    cn_model *model = cn_model_create(&config, error, sizeof(error));
    if (!model) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    int tokens[] = {CN_IM_START, 'h', 'e', 'l', 'l', 'o', CN_IM_END, CN_EOT};
    cn_optimizer optimizer = cn_optimizer_default();
    cn_metrics metrics;
    cn_set_threads(1);
    if (!cn_train_step(model, tokens, sizeof(tokens) / sizeof(tokens[0]), &optimizer, &metrics)) {
        cn_model_free(model);
        return 1;
    }
    cn_context *context = cn_context_create(model, 0);
    if (!context) {
        cn_model_free(model);
        return 1;
    }
    float logits[CN_VOCAB];
    if (!cn_prefill_ced(context, tokens, 6, 0, logits)) {
        cn_context_free(context);
        cn_model_free(model);
        return 1;
    }
    uint64_t random = 7;
    int next = cn_sample(logits, CN_VOCAB, 0, 0, &random);
    printf("parameters=%zu loss=%.4f next_token=%d\n", cn_parameter_count(model), metrics.loss, next);
    cn_context_free(context);
    cn_model_free(model);
    return 0;
}
