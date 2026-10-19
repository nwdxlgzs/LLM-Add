#ifndef DSV41_ADD_H
#define DSV41_ADD_H
#include "cairn.h"
#include "cairn_cpu.h"
#include <stdio.h>
#define ADD_LOCAL_LENGTH 8
#define ADD_PROMPT_LENGTH 5
#define ADD_DIGITS 10

typedef struct {
    int a, b, cin, digit, cout;
} add_column;
typedef struct {
    add_column columns[ADD_DIGITS];
    int length, valid;
    char answer[12];
} add_trace;
typedef struct {
    int correct, total;
    float margin;
    double probability;
} add_result;
typedef struct {
    int pretrain, sft, rl, batch;
    uint64_t seed;
} add_training;
void add_config(cn_config *c, uint64_t seed);
uint64_t add_random(uint64_t *rng);
double add_uniform(uint64_t *rng);
double add_now(void);
int add_mkdir(const char *path);
int add_path(char *out, size_t capacity, const char *dir, const char *name);
int add_parse(const char *text, uint64_t *out);
void add_example(int a, int b, int cin, int *tokens);
/* No arithmetic oracle or table in local inference: 3 input digit bytes,
   two autoregressively predicted digit bytes, then a learned end marker. */
int add_local(cn_context *context, int a, int b, int cin, add_column *out, uint64_t *sample_rng,
              float *margin, double *probability);
int add_predict(cn_context *context, const char *a, const char *b, add_trace *trace, uint64_t *rng);
int add_print(FILE *file, const char *a, const char *b, const add_trace *trace);
double add_reward(const char *a, const char *b, const add_trace *trace, int *exact);
add_result add_check(cn_model *m, FILE *details);
/* Certifies the current loaded representation without converting/repacking. */
int add_certify(cn_model *m, const char *checkpoint, const char *report, int all);
int add_evaluate(cn_model *m, const char *report, unsigned random_count, uint64_t seed);
int add_prepare(const char *dir, uint64_t seed);
add_training add_default_training(void);
int add_train(const char *dir, add_training config);
int add_qat(const char *input, const char *dir, int steps, int batch);
#endif
