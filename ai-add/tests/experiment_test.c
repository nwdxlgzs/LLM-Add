#include "add_experiment.h"
#include "add_session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "experiment FAIL %d: %s\n", __LINE__, #x);                             \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static char *read_all(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *s = malloc((size_t)n + 1);
    if (!s) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(s, 1, (size_t)n, f);
    s[got] = 0;
    fclose(f);
    return s;
}
int main(int argc, char **argv) {
    CHECK(argc == 2);
    char error[256];
    cn_model *m = cn_load(argv[1], error, sizeof(error));
    CHECK(m);
    char a[1001];
    memset(a, '9', 1000);
    a[1000] = 0;
    CHECK(add_experiment_predict(m, a, "1", 1, "long.json", "long.txt"));
    char *json = read_all("long.json");
    CHECK(json && strstr(json, "\"valid\":true"));
    char expected[1020];
    strcpy(expected, "\"answer\":\"1");
    memset(expected + 11, '0', 1000);
    expected[1011] = '"';
    expected[1012] = 0;
    CHECK(strstr(json, expected));
    free(json);
    CHECK(!add_experiment_predict(m, "12x", "1", 1, "invalid.json", "invalid.txt"));
    CHECK(add_experiment_predict(m, a, "1", 0, "direct.json", "direct.txt"));
    json = read_all("direct.json");
    CHECK(json && strstr(json, "\"mode\":\"direct\"") && !strstr(json, "\"rollovers\":0"));
    free(json);
    cn_model_free(m);
    cn_config c;
    add_config(&c, 20261018);
    c.max_seq = 64;
    add_session *s = add_session_create(&c, "experiment-test-data");
    CHECK(s && add_session_set_cot(s, 0));
    CHECK(add_experiment_predict(add_session_model(s), "99", "99", 1, "untrained.json",
                                 "untrained.txt"));
    json = read_all("untrained.json");
    CHECK(json && strstr(json, "\"valid\":false") && strstr(json, "\"ids\":["));
    free(json);
    for (int stage = 1; stage <= 6; stage++) {
        CHECK(add_session_begin(s, stage, 2, 1, 0));
        CHECK(!add_session_set_cot(s, 1));
        CHECK(add_session_tick(s) == 0);
        CHECK(add_session_tick(s) == 1);
        CHECK(add_session_progress(s)->step == 2 && !add_session_progress(s)->running);
    }
    CHECK(add_session_cot(s) == 0);
    add_session_free(s);
    puts("unbounded/failed inference and six answer-only training stages passed");
    return 0;
}
