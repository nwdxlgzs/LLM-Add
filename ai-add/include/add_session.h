#ifndef ADD_SESSION_H
#define ADD_SESSION_H
#include "add.h"
#include "add_chat.h"
typedef struct add_session add_session;
/* One tick is one attempted update. RLOO skips optimizer drift for equal rewards. */
typedef struct {
    int stage, step, total, running, metric_step, informative;
    float loss, margin;
    int correct;
    double probability, reward, seconds;
    size_t arena_bytes;
    double validation_loss;
    int token_correct, token_total, by_width[5], best_step, last_correct;
    uint64_t optimizer_step;
} add_progress;
add_session *add_session_create(const cn_config *config, const char *directory);
void add_session_free(add_session *s);
cn_model *add_session_model(add_session *s);
int add_session_load(add_session *s, const char *path);
int add_session_begin(add_session *s, int stage, int steps, int batch, float learning_rate);
/* Continue a selected stage-4 policy at a later curriculum position. Keeps
   loaded optimizer moments; the caller records the earlier attempt history.
   Does not restore a previous RNG stream. Must be called before the first tick. */
int add_session_continue_curriculum(add_session *s, int completed);
/* -1=error, 0=more updates, 1=stage complete. */
int add_session_tick(add_session *s);
add_result add_session_check(add_session *s);
const add_progress *add_session_progress(const add_session *s);
int add_session_set_cot(add_session *s, int enabled);
int add_session_cot(const add_session *s);
/* protocol: 0 legacy, 1 compact chat-v1, 2 structured column-v2,
   3 labeled whole-dialogue chat-v2. */
int add_session_set_chat(add_session *s, int enabled);
int add_session_chat(const add_session *s);
const add_chat_rl_trace *add_session_rollouts(const add_session *s);
#endif
