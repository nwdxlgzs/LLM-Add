#include "add.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                                   \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--fp4-file")) {
        char error[256];
        cn_model *m = cn_load(argv[2], error, sizeof(error));
        CHECK(m && cn_model_precision(m) == CN_MXFP4);
        CHECK(cn_parameter_count(m) == 57388);
        CHECK(cn_model_quantized_parameters(m, CN_MXFP4) == 20736);
        CHECK(cn_model_quantized_parameters(m, CN_FP8) == 30520);
        /* Drop packed matrices' FP32 copies. Certification must still work
           using the stored nibbles/bytes, including the Engram table. */
        CHECK(cn_model_freeze(m));
        CHECK(add_check(m, NULL).correct == 200);
        cn_model_free(m);
        puts("actual FP4 storage/frozen inference passed");
        return 0;
    }
    CHECK(argc == 1);
    uint64_t n;
    CHECK(add_parse("9999999999", &n) && n == UINT64_C(9999999999));
    CHECK(!add_parse("10000000000", NULL));
    CHECK(!add_parse("-1", NULL));
    cn_config c;
    add_config(&c, 20261018);
    char error[256];
    CHECK(cn_config_validate(&c, error, sizeof(error)));
    CHECK(c.layers == 6 && c.encoder_layers == 3 && c.hc == 2 && c.layer[1].engram.enabled &&
          c.experts == 2);
    cn_model *m = cn_model_create(&c, error, sizeof(error));
    CHECK(m);
    CHECK(cn_parameter_count(m) < 100000);
    printf("6-layer DSV4.1F model parameters=%zu\n", cn_parameter_count(m));
    int t[8];
    add_example(9, 8, 1, t);
    cn_metrics original, explicit_loss;
    CHECK(cn_loss_backward(m, t, 8, &original));
    cn_token_objective full[8] = {{0}};
    for (int i = 1; i < 8; i++)
        full[i].ce = 1.f / 7;
    CHECK(cn_objective_backward(m, t, 8, full, 0, 1, &explicit_loss));
    CHECK(fabsf(original.loss - explicit_loss.loss) < 2e-5f);
    cn_context *context = cn_context_create(m, 0);
    CHECK(context);
    float logits[7 * CN_VOCAB];
    CHECK(cn_prefill(context, t, 7, logits));
    cn_context_free(context);
    double expected = 0;
    memset(full, 0, sizeof(full));
    for (int i = 5; i < 8; i++) {
        full[i].ce = 1.f / 3;
        float *x = logits + (i - 1) * CN_VOCAB;
        double maximum = x[0], sum = 0;
        for (int j = 1; j < CN_VOCAB; j++)
            if (x[j] > maximum)
                maximum = x[j];
        for (int j = 0; j < CN_VOCAB; j++)
            sum += exp(x[j] - maximum);
        expected += (log(sum) + maximum - x[t[i]]) / 3;
    }
    CHECK(cn_objective_backward(m, t, 8, full, 0, 0, &explicit_loss));
    CHECK(fabs(expected - explicit_loss.loss) < 2e-5);
    cn_token_objective o[8] = {{0}};
    float reference[CN_VOCAB];
    for (int i = 0; i < CN_VOCAB; i++)
        reference[i] = sinf((float)i) * .7f;
    o[5].ce = -.7f;
    o[5].kl = .03f;
    o[5].entropy = .02f;
    o[5].reference_logits = reference;
    o[6].ce = .4f;
    o[7].ce = .2f;
    float max_error;
    CHECK(cn_objective_gradient_check(m, t, 8, o, .002f, .15f, &max_error));
    printf("policy/mask gradient max_relative_error=%g\n", (double)max_error);
    cn_metrics stats;
    CHECK(cn_objective_backward(m, t, 8, o, 0, 0, &stats));
    CHECK(isfinite(stats.loss));
    cn_optimizer optimizer = cn_optimizer_default();
    CHECK(cn_optimizer_step(m, &optimizer, &stats));
    CHECK(cn_optimizer_reset(m));
    memset(o, 0, sizeof(o));
    CHECK(cn_objective_backward(m, t, 8, o, 0, 0, &stats));
    CHECK(stats.loss == 0);
    CHECK(cn_optimizer_step(m, &optimizer, &stats));
    CHECK(stats.grad_norm == 0);
    CHECK(stats.step == 1);
    o[5].ce = NAN;
    CHECK(!cn_objective_backward(m, t, 8, o, 0, 0, &stats));
    o[5].ce = 0;
    o[5].kl = 1;
    CHECK(!cn_objective_backward(m, t, 8, o, 0, 0, &stats));
    add_trace trace = {{{9, 1, 0, 0, 1}, {9, 0, 1, 0, 1}}, 2, 1, "100"};
    int exact;
    CHECK(add_reward("99", "01", &trace, &exact) == 1 && exact);
    trace.columns[1].cin = 0;
    CHECK(add_reward("99", "01", &trace, &exact) == 0 && !exact);
    CHECK(!cn_model_set_qat(m, CN_KVFP4));
    for (int type = CN_FP8; type <= CN_MXFP4; type++) {
        CHECK(cn_model_set_qat(m, (cn_dtype)type));
        CHECK(cn_model_config(m)->qat && cn_model_precision(m) == (cn_dtype)type);
        CHECK(cn_model_quantized_parameters(m, CN_FP8) > 0);
        CHECK((cn_model_quantized_parameters(m, CN_MXFP4) > 0) == (type == CN_MXFP4));
        cn_context *inference = cn_context_create(m, 0), *training = cn_context_create(m, 1);
        CHECK(inference && training);
        double quantized_ce = 0;
        for (int i = 0; i < 7; i++) {
            const float *x, *y;
            CHECK(cn_forward(inference, t[i], &x));
            CHECK(cn_forward(training, t[i], &y));
            double maximum = x[0], sum = 0;
            for (int j = 0; j < CN_VOCAB; j++) {
                CHECK(fabsf(x[j] - y[j]) < 3e-5f);
                if (x[j] > maximum)
                    maximum = x[j];
            }
            if (i >= 4) {
                for (int j = 0; j < CN_VOCAB; j++)
                    sum += exp(x[j] - maximum);
                quantized_ce += (log(sum) + maximum - x[t[i + 1]]) / 3;
            }
        }
        cn_context_free(inference);
        cn_context_free(training);
        CHECK(cn_objective_backward(m, t, 8, full, 0, 0, &stats));
        CHECK(fabs(quantized_ce - stats.loss) < 2e-5);
        CHECK(cn_optimizer_step(m, &optimizer, &stats));
        CHECK(stats.grad_norm > 0 && isfinite(stats.grad_norm));
        /* Optimizer repacks the new masters. The next QAT and inference pass
           must still agree; packed buffers may not lag one update behind. */
        inference = cn_context_create(m, 0);
        training = cn_context_create(m, 1);
        CHECK(inference && training);
        for (int i = 0; i < 7; i++) {
            const float *x, *y;
            CHECK(cn_forward(inference, t[i], &x) && cn_forward(training, t[i], &y));
            for (int j = 0; j < CN_VOCAB; j++)
                CHECK(fabsf(x[j] - y[j]) < 3e-5f);
        }
        cn_context_free(inference);
        cn_context_free(training);
    }
    CHECK(cn_model_set_qat(m, CN_F32));
    CHECK(!cn_model_config(m)->qat && cn_model_precision(m) == CN_F32);
    CHECK(!cn_model_quantized_parameters(m, CN_FP8) && !cn_model_quantized_parameters(m, CN_MXFP4));
    CHECK(cn_model_set_qat(m, CN_MXFP4) && cn_model_freeze(m));
    CHECK(!cn_model_set_qat(m, CN_FP8));
    cn_model_free(m);
    puts("addition/objective tests passed");
    return 0;
}
