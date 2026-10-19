#ifndef ADD_EXPERIMENT_H
#define ADD_EXPERIMENT_H
#include "add.h"

/* Experimental evaluation never gates on accuracy. JSON/raw files keep even
   malformed predictions. Returns 1 for a completed attempt, 0 for I/O/OOM. */
int add_experiment_predict(cn_model *model, const char *a, const char *b, int cot,
                           const char *json_path, const char *raw_path);
/* Whole-expression answer-only baseline; no carry or reasoning targets. */
int add_direct_update(cn_model *model, cn_model *reference, int stage, int step, int total,
                      int batch, uint64_t *rng, const cn_optimizer *optimizer, cn_metrics *metrics,
                      double *reward, int *informative);
add_result add_direct_check(cn_model *model);
#endif
