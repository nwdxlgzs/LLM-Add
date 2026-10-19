#ifndef ADD_COLUMN_H
#define ADD_COLUMN_H
#include "add_chat.h"
/* Position belongs to the scheduler: arithmetic transitions are position invariant.
   Every model invocation is a real user/assistant exchange, with no think token. */
int add_column_example(int a, int b, int cin, int copy, int *tokens, int capacity, int *prompt);
int add_column_local(cn_context *ctx, int a, int b, int cin, int ids[5], uint64_t *rng,
                     float *margin);
int add_column_validate(cn_model *m, add_chat_validation *result);
int add_column_update(cn_model *m, cn_model *ref, int stage, int step, int total, int batch,
                      uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats,
                      double *reward, int *informative);
int add_column_predict(cn_model *m, const char *a, const char *b, const char *json, const char *raw);
#endif
