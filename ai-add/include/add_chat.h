#ifndef ADD_CHAT_H
#define ADD_CHAT_H
#include "add.h"
#define ADD_CHAT_CAPACITY 512
#define ADD_COMPACT_CHAT_CAPACITY 256
void add_chat_config(cn_config *config, uint64_t seed);
/* Resize all width bottlenecks together; preserve the six-layer topology.
   dim=24, ff=24 reproduces the historical tiny configuration exactly. */
int add_chat_resize(cn_config *config, int dim, int ff);
/* Six/eight/ten-layer reduced CED schedules; new experiments <=500k scalars.
   Historical larger checkpoints remain loadable. */
int add_chat_architecture(cn_config *config, int layers, int dim, int ff);
#define ADD_PARAMETER_LIMIT 500000
typedef struct {
    char a[12], b[12];
    int count[4], prompt, tokens[4][ADD_CHAT_CAPACITY];
    double reward[4];
} add_chat_rl_trace;
int add_chat_save_rollouts(const add_chat_rl_trace *trace, const char *path);
double add_chat_outcome_reward(const char *a, const char *b, int cot,
                               const int *tokens, int count);
/* Returns the selected operand width; exported so curriculum coverage is testable. */
int add_chat_problem(int stage, int step, int total, uint64_t *rng, char *a, char *b);
/* Exactly one user/assistant exchange. Only assistant content is supervised.
   CoT is one optional block in the assistant answer, never a column prompt. */
int add_chat_example(const char *a, const char *b, int cot, int copy, int *tokens, int capacity,
                     int *prompt);
int add_chat_predict(cn_model *m, const char *a, const char *b, int cot, const char *json_path,
                     const char *raw_path);
int add_chat_update(cn_model *m, cn_model *ref, int stage, int step, int total, int batch, int cot,
                    uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward,
                    int *informative);
add_result add_chat_check(cn_model *m, int cot);
typedef struct {
    add_result exact;
    double loss;
    int token_correct, token_total, by_width[5];
} add_chat_validation;
int add_chat_validate_outcome(cn_model *model, int cot, add_chat_validation *result);
int add_chat_update_traced(cn_model *m, cn_model *ref, int stage, int step, int total, int batch,
                           int cot, uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats,
                           double *reward, int *informative, int fields, add_chat_rl_trace *trace);
int add_chat_validate(cn_model *m, int cot, add_chat_validation *result);
/* fields=1: chat-v2 labeled scratchpad; fields=0: historical compact chat-v1.
   Format is explicit metadata, never inferred from a generated answer. */
int add_chat_example_format(const char *a, const char *b, int cot, int copy, int *tokens,
                            int capacity, int *prompt, int fields);
int add_chat_predict_format(cn_model *m, const char *a, const char *b, int cot,
                            const char *json_path, const char *raw_path, int fields);
int add_chat_update_format(cn_model *m, cn_model *ref, int stage, int step, int total, int batch,
                           int cot, uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats,
                           double *reward, int *informative, int fields);
int add_chat_validate_format(cn_model *m, int cot, add_chat_validation *result, int fields);
double add_chat_reward(const char *a, const char *b, int cot, const int *tokens, int count,
                       int fields);
#endif
