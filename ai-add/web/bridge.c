#include "add_chat.h"
#include "add_column.h"
#include "add_experiment.h"
#include "add_session.h"
#include <emscripten/emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static add_session *session;
static cn_model *deployed;
static int deployed_ok, student_ok;
static int deployed_chat, cache_enabled = 1;
static char output[12000];
static char *prediction_json;

EMSCRIPTEN_KEEPALIVE int web_init_arch(int dim, int ff, int layers, unsigned seed) {
    cn_config c;
    add_chat_config(&c, seed);
    if (!add_chat_architecture(&c, layers, dim, ff)) return 0;
    c.max_seq = ADD_CHAT_CAPACITY;
    add_session *next = add_session_create(&c, "/experiment");
    if (!next)
        return 0;
    add_session_free(session);
    session = next;
    add_session_set_chat(session, 3);
    student_ok = 0;
    return 1;
}
EMSCRIPTEN_KEEPALIVE int web_init(int dim, int ff, unsigned seed) {
    return web_init_arch(dim,ff,6,seed);
}
EMSCRIPTEN_KEEPALIVE int web_shape(int dim, int ff, int layers) {
    cn_config c;add_chat_config(&c,1);
    if(!add_chat_architecture(&c,layers,dim,ff))return 0;
    char error[256];cn_model *m=cn_model_create(&c,error,sizeof(error));
    if(!m)return 0;int n=(int)cn_parameter_count(m);cn_model_free(m);return n;
}
EMSCRIPTEN_KEEPALIVE int web_rollouts(void) {
    return add_chat_save_rollouts(add_session_rollouts(session),"/rollouts.json");
}
EMSCRIPTEN_KEEPALIVE double web_outcome_reward(const char *a,const char *b,int cot,const char *text) {
    int tokens[ADD_CHAT_CAPACITY];
    size_t n=cn_encode(text,strlen(text),1,tokens,ADD_CHAT_CAPACITY);
    if(n==SIZE_MAX || n>ADD_CHAT_CAPACITY)return 0;
    return add_chat_outcome_reward(a,b,cot,tokens,(int)n);
}
EMSCRIPTEN_KEEPALIVE int web_load(const char *path, int student) {
    if (student) {
        int ok = add_session_load(session, path);
        if (ok)
            student_ok = 0;
        return ok;
    }
    char error[256];
    cn_model *next = cn_load(path, error, sizeof(error));
    if (!next)
        return 0;
    cn_model_free(deployed);
    deployed = next;
    deployed_chat = 0;
    deployed_ok = 0;
    return 1;
}
EMSCRIPTEN_KEEPALIVE int web_begin(int stage, int steps, int batch, float lr) {
    student_ok = 0;
    return add_session_begin(session, stage, steps, batch, lr);
}
EMSCRIPTEN_KEEPALIVE int web_mode(int enabled) { return add_session_set_cot(session, enabled); }
EMSCRIPTEN_KEEPALIVE int web_protocol(int student, int chat) {
    if (student)
        return add_session_set_chat(session, chat);
    if (chat < 0 || chat > 3)
        return 0;
    deployed_chat = chat;
    return 1;
}
EMSCRIPTEN_KEEPALIVE int web_cache(int student, int enabled) {
    cn_model *m = student ? add_session_model(session) : deployed;
    if (!cn_model_inference_cache(m, enabled))
        return 0;
    cache_enabled = enabled;
    return 1;
}
EMSCRIPTEN_KEEPALIVE const char *web_example_format(const char *a, const char *b, int cot, int copy, int fields) {
    int tokens[ADD_CHAT_CAPACITY], prompt = 0;
    int n = add_chat_example_format(a, b, cot, copy, tokens, ADD_CHAT_CAPACITY, &prompt, fields);
    if (!n)
        return "{}";
    int used = snprintf(output, sizeof(output), "{\"prompt\":%d,\"tokens\":[", prompt);
    for (int i = 0; i < n; i++)
        used +=
            snprintf(output + used, sizeof(output) - (size_t)used, "%s%d", i ? "," : "", tokens[i]);
    snprintf(output + used, sizeof(output) - (size_t)used, "]}");
    return output;
}
EMSCRIPTEN_KEEPALIVE const char *web_example(const char *a, const char *b, int cot, int copy) {
    return web_example_format(a,b,cot,copy,1);
}
EMSCRIPTEN_KEEPALIVE const char *web_column_example(int a, int b, int cin, int copy) {
    int tokens[128], prompt=0;
    int n=add_column_example(a,b,cin,copy,tokens,128,&prompt);
    if(!n)return "{}";
    int used=snprintf(output,sizeof(output),"{\"prompt\":%d,\"tokens\":[",prompt);
    for(int i=0;i<n;i++)used+=snprintf(output+used,sizeof(output)-(size_t)used,"%s%d",i?",":"",tokens[i]);
    snprintf(output+used,sizeof(output)-(size_t)used,"]}");return output;
}
EMSCRIPTEN_KEEPALIVE int web_tick(void) {
    student_ok = 0;
    return add_session_tick(session);
}
EMSCRIPTEN_KEEPALIVE const char *web_status(int student) {
    cn_model *m = student ? add_session_model(session) : deployed;
    if (!m)
        return "{}";
    cn_cpu_info cpu;
    cn_cpu_detect(&cpu);
    const cn_config *c = cn_model_config(m);
    const add_progress *p = add_session_progress(session);
    snprintf(output, sizeof(output),
             "{\"parameters\":%zu,\"dim\":%d,\"ff\":%d,\"layers\":%d,\"encoder\":%d,\"window\":%d,"
             "\"context\":%d,\"heads\":%d,\"headDim\":%d,\"qRank\":%d,\"oRank\":%d,\"ropeDim\":%d,\"indexDim\":%d,\"engramDim\":%d,"
             "\"precision\":%d,\"fp4\":%zu,\"fp8\":%zu,\"backend\":\"%s\","
             "\"certified\":%s,\"stage\":%d,\"step\":%d,\"total\":%d,\"running\":%d,"
             "\"metricStep\":%d,\"correct\":%d,\"margin\":%.9g,\"loss\":%.9g,"
             "\"probability\":%.9g,\"reward\":%.9g,\"informative\":%d,\"seconds\":%.6f,"
             "\"arena\":%zu,\"cot\":%s,\"protocol\":\"%s\",\"cache_bytes\":%zu,"
             "\"validationLoss\":%.9g,\"tokenCorrect\":%d,\"tokenTotal\":%d,"
             "\"widthCorrect\":[%d,%d,%d,%d,%d],\"bestStep\":%d,\"lastCorrect\":%d,\"optimizerStep\":%llu}",
             cn_parameter_count(m), c->dim, c->ff_dim, c->layers, c->encoder_layers, c->window,
             c->max_seq,c->heads,c->head_dim,c->q_rank,c->o_rank,c->rope_dim,c->index_dim,c->layer[1].engram.head_dim,
             cn_model_precision(m), cn_model_quantized_parameters(m, CN_MXFP4),
             cn_model_quantized_parameters(m, CN_FP8), cn_cpu_backend_name(cpu.selected),
             (student ? student_ok : deployed_ok) ? "true" : "false", p->stage, p->step, p->total,
             p->running, p->metric_step, p->correct, (double)p->margin, (double)p->loss,
             p->probability, p->reward, p->informative, p->seconds, p->arena_bytes,
             add_session_cot(session) ? "true" : "false",
             (student ? add_session_chat(session) : deployed_chat) == 3 ? "chat-v2" :
             (student ? add_session_chat(session) : deployed_chat) == 2 ? "column-v2" :
             (student ? add_session_chat(session) : deployed_chat) == 1 ? "chat-v1" : "legacy-v1",
             cn_model_inference_cache_bytes(m), p->validation_loss, p->token_correct, p->token_total,
             p->by_width[0],p->by_width[1],p->by_width[2],p->by_width[3],p->by_width[4],
             p->best_step,p->last_correct,(unsigned long long)p->optimizer_step);
    return output;
}
EMSCRIPTEN_KEEPALIVE int web_certify(int student) {
    cn_model *m = student ? add_session_model(session) : deployed;
    if (!m)
        return 0;
    /* Save/load the actual storage representation before certifying it. */
    if (!cn_save(m, "/candidate.cn", 0))
        return 0;
    char error[256];
    cn_model *candidate = cn_load("/candidate.cn", error, sizeof(error));
    if (!candidate)
        return 0;
    int ok = add_certify(candidate, "/candidate.cn", "/certificate.json", 1);
    cn_model_free(candidate);
    if (student) {
        student_ok = ok;
        add_session_check(session);
    } else
        deployed_ok = ok;
    return ok;
}
EMSCRIPTEN_KEEPALIVE int web_predict(const char *a, const char *b, int student, int cot) {
    cn_model *m = student ? add_session_model(session) : deployed;
    if (!cn_model_inference_cache(m, cache_enabled))
        return 0;
    int protocol=student ? add_session_chat(session) : deployed_chat;
    return protocol == 2 && cot ? add_column_predict(m,a,b,"/prediction.json","/inference.txt")
         : protocol ? add_chat_predict_format(m, a, b, cot, "/prediction.json", "/inference.txt",protocol==3)
                    : add_experiment_predict(m, a, b, cot, "/prediction.json", "/inference.txt");
}
EMSCRIPTEN_KEEPALIVE const char *web_infer(const char *a, const char *b, int student) {
    if (!web_predict(a, b, student, 1))
        return "{\"ok\":false,\"error\":\"input_or_memory\"}";
    FILE *f = fopen("/prediction.json", "rb");
    if (!f)
        return "{}";
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *next = size >= 0 ? malloc((size_t)size + 1) : NULL;
    if (!next) {
        fclose(f);
        return "{}";
    }
    size_t count = fread(next, 1, (size_t)size, f);
    fclose(f);
    next[count] = 0;
    free(prediction_json);
    prediction_json = next;
    return next;
}
EMSCRIPTEN_KEEPALIVE int web_save(const char *path, int optimizer) {
    return session && cn_save(add_session_model(session), path, optimizer);
}
EMSCRIPTEN_KEEPALIVE int web_evaluate(int student, int random_count) {
    if (random_count < 0 || random_count > 10000)
        return 0;
    cn_model *m = student ? add_session_model(session) : deployed;
    return m && add_evaluate(m, "/evaluation.json", (unsigned)random_count, 20261019);
}
