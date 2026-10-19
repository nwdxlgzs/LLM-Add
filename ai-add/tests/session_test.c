#include "add_session.h"
#include <math.h>
#include <stdio.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "session FAIL %d: %s\n", __LINE__, #x);                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(int argc, char **argv) {
    CHECK(argc == 2);
    cn_config c;
    add_config(&c, 20261018);
    add_session *s = add_session_create(&c, "session-test-data");
    CHECK(s && cn_parameter_count(add_session_model(s)) == 57388);
    CHECK(!add_session_begin(s, 7, 1, 4, 0));
    CHECK(!add_session_begin(s, 1, 0, 4, 0));
    for (int stage = 1; stage <= 6; stage++) {
        CHECK(add_session_begin(s, stage, 2, 4, 0));
        CHECK(!add_session_begin(s, stage, 2, 4, 0));
        CHECK(!add_session_load(s, argv[1]));
        CHECK(add_session_tick(s) == 0);
        CHECK(add_session_tick(s) == 1);
        const add_progress *p = add_session_progress(s);
        CHECK(!p->running && p->step == 2 && p->metric_step == 2 && isfinite(p->loss));
        CHECK(cn_parameter_count(add_session_model(s)) == 57388);
    }
    CHECK(cn_model_precision(add_session_model(s)) == CN_MXFP4);
    CHECK(cn_model_quantized_parameters(add_session_model(s), CN_MXFP4) == 20736);
    CHECK(add_session_load(s, argv[1]));
    CHECK(add_session_check(s).correct == 200);
    add_session_free(s);
    puts("incremental pretrain/SFT/RLOO/QAT sessions passed");
    return 0;
}
