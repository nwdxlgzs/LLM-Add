#include "add_chat.h"
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void add_chat_config(cn_config *c, uint64_t seed) {
    add_config(c, seed);
    c->max_seq = ADD_CHAT_CAPACITY;
    /* Keep the entire ten-digit dialogue visible, including the question when
       emitting its answer. The old 64-token window dropped input digits during
       CoT; sparse top-k memory alone is a poor copying curriculum. */
    c->window = ADD_CHAT_CAPACITY;
}
int add_chat_resize(cn_config *c, int dim, int ff) {
    if (!c || dim < 16 || dim > 128 || dim % 8 || ff < 16 || ff > 512 || ff % 8)
        return 0;
    /* Keep two heads, with per-head width approximately dim/3, aligned to
       eight. Also widen the low-rank projections, indexer and Engram. */
    int head = 8 * ((dim + 12) / 24);
    if (head < 8) head = 8;
    c->dim = dim;
    c->ff_dim = ff;
    c->head_dim = c->q_rank = c->o_rank = head;
    c->rope_dim = head / 2;
    c->index_dim = head / 2;
    c->layer[1].engram.head_dim = head / 4;
    char error[256];
    return cn_config_validate(c, error, sizeof(error));
}
int add_chat_architecture(cn_config *c, int layers, int dim, int ff) {
    if (!c || (layers != 6 && layers != 8 && layers != 10)) return 0;
    cn_config next = *c;
    if (!add_chat_resize(&next, dim, ff)) return 0;
    cn_engram_config engram = next.layer[1].engram;
    memset(next.layer, 0, sizeof(next.layer));
    next.layers = layers;
    next.encoder_layers = layers / 2;
    for (int i=0; i<CN_MAX_LAYERS; i++) {
        next.layer[i].kv_source = next.layer[i].index_source = -1;
        next.layer[i].compression = 1;
    }
    next.layer[1].engram = engram;
    for (int i=1; i<layers; i++) {
        int owner = i < next.encoder_layers ? 1 : next.encoder_layers;
        next.layer[i].kv_source = owner;
        next.layer[i].index_source = i == layers-1 ? i : owner;
        next.layer[i].compression = i < next.encoder_layers ? 2 : 1;
    }
    char error[256];
    if (!cn_config_validate(&next,error,sizeof(error))) return 0;
    cn_model *probe = cn_model_create(&next,error,sizeof(error));
    if (!probe) return 0;
    size_t count = cn_parameter_count(probe);
    cn_model_free(probe);
    if (count > ADD_PARAMETER_LIMIT) return 0;
    *c = next;
    return 1;
}

static int decimal(const char *p) {
    if (!p || !*p)
        return 0;
    for (; *p; p++)
        if (*p < '0' || *p > '9')
            return 0;
    return 1;
}
static int best(const float *x) {
    int b = 0;
    for (int i = 0; i < CN_VOCAB; i++) {
        if (!isfinite(x[i]))
            return -1;
        if (x[i] > x[b])
            b = i;
    }
    return b;
}
static void string_json(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\')
            fprintf(f, "\\%c", *p);
        else if (*p < 32 || *p >= 127)
            fprintf(f, "\\u%04x", *p);
        else
            fputc(*p, f);
    }
    fputc('"', f);
}
static void ids_json(FILE *f, const int *ids, size_t n) {
    fputc('[', f);
    for (size_t i = 0; i < n; i++)
        fprintf(f, "%s%d", i ? "," : "", ids[i]);
    fputc(']', f);
}
int add_chat_save_rollouts(const add_chat_rl_trace *t, const char *path) {
    if(!t || !path) return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    fprintf(f,"{\"a\":\"%s\",\"b\":\"%s\",\"prompt\":%d,\"samples\":[",t->a,t->b,t->prompt);
    for(int k=0;k<4;k++) {
        fprintf(f,"%s{\"reward\":%.9g,\"tokens\":",k?",":"",t->reward[k]);
        ids_json(f,t->tokens[k],(size_t)t->count[k]);fputc('}',f);
    }
    fputs("]}",f);int ok=!ferror(f);if(fclose(f))ok=0;return ok;
}
static char *prompt_text(const char *a, const char *b, int copy) {
    size_t na = strlen(a), nb = strlen(b);
    if (na > SIZE_MAX - nb - 128)
        return NULL;
    char *text = malloc(na + nb + 128);
    if (text)
        snprintf(text, na + nb + 128,
                 copy ? "<|im_start|>user\nCopy: %s%s<|im_end|>\n<|im_start|>assistant\n"
                      : "<|im_start|>user\n%s+%s<|im_end|>\n<|im_start|>assistant\n",
                 a, copy ? "" : b);
    return text;
}
int add_chat_example_format(const char *a, const char *b, int cot, int copy, int *t, int capacity,
                            int *prompt, int fields) {
    if (!decimal(a) || !decimal(b) || !t || capacity < 1 || !prompt)
        return 0;
    uint64_t va, vb;
    if (!add_parse(a, &va) || !add_parse(b, &vb))
        return 0;
    char *input = prompt_text(a, b, copy);
    if (!input)
        return 0;
    size_t p = cn_encode(input, strlen(input), 1, t, (size_t)capacity);
    free(input);
    if (p == SIZE_MAX || p >= (size_t)capacity)
        return 0;
    char body[1024];
    size_t n = 0;
    if (copy)
        n = (size_t)snprintf(body, sizeof(body), "%s", a);
    else {
        if (cot) {
            n = (size_t)snprintf(body, sizeof(body), "<think>\n");
            size_t na = strlen(a), nb = strlen(b), width = na > nb ? na : nb;
            int carry = 0;
            for (size_t i = 0; i < width; i++) {
                int da = i < na ? a[na - 1 - i] - '0' : 0;
                int db = i < nb ? b[nb - 1 - i] - '0' : 0, sum = da + db + carry;
                if(fields)
                    n += (size_t)snprintf(body + n, sizeof(body) - n,
                                          "pos=%zu a=%d b=%d cin=%d -> %d %d\n",
                                          i, da, db, carry, sum % 10, sum / 10);
                else
                    n += (size_t)snprintf(body + n, sizeof(body) - n, "%d+%d+%d=%d,%d\n",
                                          da, db, carry, sum % 10, sum / 10);
                carry = sum / 10;
            }
            n += (size_t)snprintf(body + n, sizeof(body) - n, "</think>\n");
        }
        n += (size_t)snprintf(body + n, sizeof(body) - n, "%" PRIu64, va + vb);
    }
    n += (size_t)snprintf(body + n, sizeof(body) - n, "<|im_end|><|endoftext|>");
    size_t tail = cn_encode(body, n, 1, t + p, (size_t)capacity - p);
    if (tail == SIZE_MAX || tail > (size_t)capacity - p)
        return 0;
    *prompt = (int)p;
    return (int)(p + tail);
}
/* A rolling context accepts longer test inputs; it does not manufacture a
   result or retain tokens outside the window. All rollovers are reported. */
typedef struct {
    cn_context *s;
    int *history, n, cap, rolls;
    const float *x;
} Stream;
static int push(Stream *s, int token) {
    if (s->n == s->cap) {
        int keep = s->cap / 2;
        memmove(s->history, s->history + s->n - keep, (size_t)keep * sizeof(int));
        s->n = keep;
        cn_context_reset(s->s);
        for (int i = 0; i < keep; i++)
            if (!cn_forward(s->s, s->history[i], &s->x))
                return 0;
        s->rolls++;
    }
    if (!cn_forward(s->s, token, &s->x))
        return 0;
    s->history[s->n++] = token;
    return 1;
}
static void token_text(FILE *f, int id) {
    char text[64];
    size_t n = cn_decode(&id, 1, 1, text, sizeof(text));
    if (n == SIZE_MAX || n > sizeof(text))
        return;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n' || (c >= 32 && c < 127))
            fputc(c, f);
        else
            fprintf(f, "\\x%02x", c);
    }
}
static const char *parse(const int *t, size_t n, int cot, char *answer, char *reason) {
    answer[0] = reason[0] = 0;
    if (n < 2 || t[n - 1] != CN_EOT || t[n - 2] != CN_IM_END)
        return "chat_end";
    size_t i = 0, out = 0;
    if (cot) {
        if (t[i++] != CN_THINK)
            return "think_start";
        while (i < n - 2 && t[i] != CN_THINK_END) {
            if (t[i] < 1 || t[i] > 255)
                return "think_format";
            reason[out++] = (char)t[i++];
        }
        reason[out] = 0;
        /* Only message/control syntax is constrained. The scratchpad's
           mathematics, wording and structure are never validated here. */
        if (strstr(reason,"/think") || strstr(reason,"<think") ||
            strstr(reason,"<|im_") || strstr(reason,"<|endoftext"))
            return "scratchpad_control";
        for (size_t j=0; j<out; j++)
            if (((unsigned char)reason[j]<32 && reason[j]!='\n' && reason[j]!='\t' && reason[j]!='\r') || reason[j]==127)
                return "scratchpad_control";
        if (i >= n - 2 || t[i++] != CN_THINK_END)
            return "think_end";
    }
    size_t end=n-2;
    while(end>i && (t[end-1]=='\n' || t[end-1]==' ' || t[end-1]=='\t' || t[end-1]=='\r'))end--;
    while (i < end && (t[i] == '\n' || t[i] == ' ' || t[i]=='\t' || t[i]=='\r'))
        i++;
    out = 0;
    for (; i < end; i++) {
        if (t[i] < '0' || t[i] > '9')
            return "answer_token";
        answer[out++] = (char)t[i];
    }
    answer[out] = 0;
    return out ? "" : "empty_answer";
}
int add_chat_predict_format(cn_model *m, const char *a, const char *b, int cot, const char *json_path,
                            const char *raw_path, int fields) {
    if (!m || !decimal(a) || !decimal(b))
        return 0;
    size_t width = strlen(a) > strlen(b) ? strlen(a) : strlen(b);
    size_t per_column = fields ? 40 : 12;
    if (width > (SIZE_MAX / sizeof(int) - 64) / per_column)
        return 0;
    char *text = prompt_text(a, b, 0);
    size_t plen = text ? cn_encode(text, strlen(text), 1, NULL, 0) : SIZE_MAX;
    size_t budget = cot ? per_column * width + 32 : width + 8;
    if (cot && budget < ADD_CHAT_CAPACITY) budget = ADD_CHAT_CAPACITY;
    int *input = plen != SIZE_MAX ? malloc(plen * sizeof(int)) : NULL;
    int *output = malloc(budget * sizeof(int));
    char *answer = calloc(budget + 1, 1), *reason = calloc(budget + 1, 1);
    Stream stream = {0};
    stream.cap = cn_model_config(m)->max_seq;
    stream.s = cn_context_create(m, 0);
    stream.history = malloc((size_t)stream.cap * sizeof(int));
    FILE *raw = fopen(raw_path, "wb"), *json = fopen(json_path, "wb");
    int ok = input && output && answer && reason && stream.s && stream.history && raw && json;
    if (ok) {
        cn_encode(text, strlen(text), 1, input, plen);
        fputs(text, raw);
        int forward = 1;
        for (size_t i = 0; forward && i < plen; i++)
            forward = push(&stream, input[i]);
        size_t generated = 0;
        int ended = 0;
        while (forward && generated < budget) {
            int id = best(stream.x);
            if (id < 0) {
                forward = 0;
                break;
            }
            output[generated++] = id;
            token_text(raw, id);
            if (id == CN_EOT) {
                ended = 1;
                break;
            }
            forward = push(&stream, id);
        }
        const char *error = !forward ? "forward"
                            : !ended ? "generation_limit"
                                     : parse(output, generated, cot, answer, reason);
        fprintf(
            json,
            "{\"ok\":true,\"protocol\":\"%s\",\"mode\":\"%s\",\"columns\":[],\"input_ids\":",
            fields ? "chat-v2" : "chat-v1", cot ? "cot" : "direct");
        ids_json(json, input, plen);
        fputs(",\"tokens\":", json);
        ids_json(json, output, generated);
        fprintf(json, ",\"valid\":%s,\"generated\":%zu,\"rollovers\":%d,\"message_complete\":%s,\"stop\":\"%s\",\"error\":",
                !*error ? "true" : "false", generated, stream.rolls, ended ? "true" : "false",
                !forward ? "forward" : ended ? "endoftext" : "generation_limit");
        string_json(json, error);
        fputs(",\"answer\":", json);
        string_json(json, answer);
        fputs(",\"reasoning\":", json);
        string_json(json, reason);
        fputc('}', json);
        ok = !ferror(raw) && !ferror(json);
    }
    if (raw && fclose(raw))
        ok = 0;
    if (json && fclose(json))
        ok = 0;
    cn_context_free(stream.s);
    free(stream.history);
    free(input);
    free(output);
    free(text);
    free(answer);
    free(reason);
    return ok;
}
int add_chat_problem(int stage, int step, int total, uint64_t *rng, char *a, char *b) {
    const int widths[] = {1, 2, 4, 6, 10};
    int w;
    if (stage == 1) w = widths[add_random(rng) % 5];
    else if (stage == 2) w = 1;
    else if (stage == 3) w = step <= total / 3 || add_random(rng) % 3 == 0 ? 1 : 2;
    else {
        /* Unlock five lengths during the first half, then mix all lengths.
           QAT/RL use all lengths immediately. */
        int unlocked = stage == 4 && total > 1 ? 1 + (int)((uint64_t)(step - 1) * 10 / (unsigned)total) : 5;
        if (unlocked > 5) unlocked = 5;
        int level = stage == 4 && unlocked < 5 && add_random(rng) % 4 ? unlocked - 1 : (int)(add_random(rng) % (unsigned)unlocked);
        w = widths[level];
    }
    uint64_t power = 1;
    for (int i = 0; i < w; i++)
        power *= 10;
    uint64_t va = add_random(rng) % power, vb = add_random(rng) % power;
    if (stage == 2)
        vb %= 10 - va;
    else if (stage > 1 && add_random(rng) % 8 == 0) {
        va = power - 1;
        vb = 1;
    }
    snprintf(a, 12, "%" PRIu64, va % UINT64_C(10000000000));
    snprintf(b, 12, "%" PRIu64, vb % UINT64_C(10000000000));
    return w;
}
double add_chat_reward(const char *a, const char *b, int cot, const int *t, int n, int fields) {
    char answer[ADD_CHAT_CAPACITY + 1], reason[ADD_CHAT_CAPACITY + 1];
    if(!t || n<0 || n>ADD_CHAT_CAPACITY) return 0;
    if (*parse(t, (size_t)n, cot, answer, reason))
        return 0;
    uint64_t va = 0, vb = 0;
    add_parse(a, &va);
    add_parse(b, &vb);
    char expected[24];
    snprintf(expected, sizeof(expected), "%" PRIu64, va + vb);
    double exact = !strcmp(answer, expected);
    if (!cot)
        return exact;
    int correct = 0, count = 0, carry = 0;
    const char *p = reason;
    size_t na = strlen(a), nb = strlen(b), width = na > nb ? na : nb;
    while (*p == '\n')
        p++;
    while (*p && count < (int)width) {
        int pos=count, da, db, ci, d, co, len = 0;
        int parsed=fields ? sscanf(p,"pos=%2d a=%1d b=%1d cin=%1d -> %1d %1d%n",&pos,&da,&db,&ci,&d,&co,&len)
                          : sscanf(p,"%1d+%1d+%1d=%1d,%1d%n",&da,&db,&ci,&d,&co,&len);
        if (parsed != (fields?6:5) || len < 1)
            break;
        int ea = (size_t)count < na ? a[na - 1 - (size_t)count] - '0' : 0;
        int eb = (size_t)count < nb ? b[nb - 1 - (size_t)count] - '0' : 0;
        correct += pos==count && da == ea && db == eb && ci == carry && d >= 0 && d <= 9 && co >= 0 && co <= 1 &&
                   d + 10 * co == da + db + ci;
        carry = co;
        count++;
        p += len;
        if (*p == '\n')
            p++;
        else
            break;
    }
    return .5 * exact + .5 * correct / width;
}
double add_chat_outcome_reward(const char *a, const char *b, int cot, const int *t, int n) {
    char answer[ADD_CHAT_CAPACITY+1], reason[ADD_CHAT_CAPACITY+1], expected[24];
    uint64_t va, vb;
    if (!t || n<0 || n>ADD_CHAT_CAPACITY || !add_parse(a,&va) || !add_parse(b,&vb) ||
        *parse(t,(size_t)n,cot,answer,reason)) return 0;
    snprintf(expected,sizeof(expected),"%" PRIu64,va+vb);
    const char *p=answer;
    while (*p=='0' && p[1]) p++;
    return !strcmp(p,expected);
}
int add_chat_update_traced(cn_model *m, cn_model *ref, int stage, int step, int total, int batch, int cot,
                           uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward,
                           int *informative, int fields, add_chat_rl_trace *trace) {
    memset(stats, 0, sizeof(*stats));
    *reward = 0;
    *informative = 0;
    int ok = 1, sequences = 0;
    if (stage != 5 && stage != 7) {
        for (int k = 0; k < batch; k++) {
            char a[12], b[12];
            int t[ADD_CHAT_CAPACITY], prompt;
            add_chat_problem(stage, step, total, rng, a, b);
            int n = add_chat_example_format(a, b, cot, stage == 1, t, ADD_CHAT_CAPACITY, &prompt, fields);
            if (!n || n > cn_model_config(m)->max_seq)
                return 0;
            cn_token_objective objective[ADD_CHAT_CAPACITY] = {{0}};
            /* Digits carry the task; avoid letting easy punctuation dominate
               the response objective. Still supervise all assistant tokens. */
            float weight = 0;
            for (int j = prompt; j < n; j++)
                weight += t[j] >= '0' && t[j] <= '9' ? 3.f : 1.f;
            for (int j = prompt; j < n; j++)
                objective[j].ce = (t[j] >= '0' && t[j] <= '9' ? 3.f : 1.f) / (weight * batch);
            cn_metrics one;
            if (!cn_objective_backward(m, t, (size_t)n, objective, k != 0, 1.f / batch, &one))
                return 0;
            stats->loss += one.loss;
            stats->index_loss += one.index_loss;
            if (one.arena_bytes > stats->arena_bytes)
                stats->arena_bytes = one.arena_bytes;
        }
    } else {
        char a[12], b[12];
        int expected[ADD_CHAT_CAPACITY], prompt;
        add_chat_problem(stage, step, total, rng, a, b);
        int n;
        if (stage==7) {
            char *input=prompt_text(a,b,0);
            if (!input) return 0;
            size_t length=cn_encode(input,strlen(input),1,expected,ADD_CHAT_CAPACITY);
            free(input);
            if(length==SIZE_MAX || length>=ADD_CHAT_CAPACITY) return 0;
            n=prompt=(int)length;
        } else n = add_chat_example_format(a, b, cot, 0, expected, ADD_CHAT_CAPACITY, &prompt, fields);
        if (!n || n > cn_model_config(m)->max_seq)
            return 0;
        int sampled[4][ADD_CHAT_CAPACITY], counts[4];
        double rewards[4], sum = 0;
        if (!ref) return 0;
        if (trace) { memset(trace,0,sizeof(*trace));strcpy(trace->a,a);strcpy(trace->b,b);trace->prompt=prompt; }
        cn_context *actor = cn_context_create(m, 0), *reference = cn_context_create(ref, 0);
        if (!actor || !reference) {
            cn_context_free(actor);
            cn_context_free(reference);
            return 0;
        }
        for (int k = 0; k < 4 && ok; k++) {
            memcpy(sampled[k], expected, (size_t)prompt * sizeof(int));
            counts[k] = prompt;
            cn_context_reset(actor);
            const float *x = NULL;
            for (int j = 0; j < prompt; j++)
                if (!cn_forward(actor, expected[j], &x)) {
                    ok = 0;
                    break;
                }
            int cap = cn_model_config(m)->max_seq;
            if (cap > ADD_CHAT_CAPACITY) cap = ADD_CHAT_CAPACITY;
            int limit = stage == 7 ? cap : (n + 16 < cap ? n + 16 : cap);
            while (ok && counts[k] < limit) {
                int id = cn_sample(x, CN_VOCAB, 1, 0, rng);
                if (id < 0) {
                    ok = 0;
                    break;
                }
                sampled[k][counts[k]++] = id;
                if (id == CN_EOT)
                    break;
                if (!cn_forward(actor, id, &x)) {
                    ok = 0;
                    break;
                }
            }
            rewards[k] = stage == 7 ? add_chat_outcome_reward(a,b,cot,sampled[k]+prompt,counts[k]-prompt)
                                    : add_chat_reward(a, b, cot, sampled[k] + prompt, counts[k] - prompt, fields);
            if (trace) { trace->count[k]=counts[k];trace->reward[k]=rewards[k];
                memcpy(trace->tokens[k],sampled[k],(size_t)counts[k]*sizeof(int)); }
            sum += rewards[k];
        }
        float *logits = malloc((size_t)ADD_CHAT_CAPACITY * CN_VOCAB * sizeof(float));
        if (!logits)
            ok = 0;
        double spread = 0;
        for (int k = 0; k < 4 && ok; k++)
            spread += fabs(rewards[k] - sum / 4);
        /* Identical rewards contain no RLOO learning signal. Do not turn this
           into entropy-only AdamW updates that drift away from the SFT policy. */
        if (ok && spread < 1e-9) {
            *reward = sum / 4;
            free(logits);
            cn_context_free(actor);
            cn_context_free(reference);
            return 1;
        }
        for (int k = 0; k < 4 && ok; k++) {
            float advantage = (float)(rewards[k] - (sum - rewards[k]) / 3);
            if (fabsf(advantage) > 1e-6f)
                *informative = 1;
            cn_context_reset(reference);
            if (!cn_prefill(reference, sampled[k], (size_t)counts[k] - 1, logits)) {
                ok = 0;
                break;
            }
            cn_token_objective objective[ADD_CHAT_CAPACITY] = {{0}};
            for (int j = prompt; j < counts[k]; j++) {
                objective[j].ce = advantage / 4;
                objective[j].kl = .001f / 4;
                objective[j].entropy = .00001f / 4;
                objective[j].reference_logits = logits + (j - 1) * CN_VOCAB;
            }
            cn_metrics one;
            ok = cn_objective_backward(m, sampled[k], (size_t)counts[k], objective,
                                       sequences++ != 0, 0, &one);
            stats->loss += one.loss;
            if (one.arena_bytes > stats->arena_bytes)
                stats->arena_bytes = one.arena_bytes;
        }
        free(logits);
        cn_context_free(actor);
        cn_context_free(reference);
        *reward = sum / 4;
    }
    return ok && cn_optimizer_step(m, opt, stats);
}
int add_chat_update_format(cn_model *m, cn_model *ref, int stage, int step, int total, int batch, int cot,
                           uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward,
                           int *informative, int fields) {
    return add_chat_update_traced(m,ref,stage,step,total,batch,cot,rng,opt,stats,reward,informative,fields,NULL);
}
int add_chat_validate_outcome(cn_model *m, int cot, add_chat_validation *r) {
    if (!m || !r) return 0;
    memset(r,0,sizeof(*r)); r->exact.total=200;
    cn_context *s=cn_context_create(m,0);
    if (!s) return 0;
    uint64_t rng=20261019;
    const int widths[]={1,2,4,6,10};
    int cap=cn_model_config(m)->max_seq;
    if (cap>ADD_CHAT_CAPACITY) cap=ADD_CHAT_CAPACITY;
    for (int i=0;i<200;i++) {
        int level=i/40,t[ADD_CHAT_CAPACITY],prompt;
        uint64_t power=1;
        for(int j=0;j<widths[level];j++) power*=10;
        uint64_t va=add_random(&rng)%power,vb=add_random(&rng)%power;
        if(i%40==0) va=vb=0;
        if(i%40==1){va=power-1;vb=1;}
        if(i%40==2) va=vb=power-1;
        char a[12],b[12];
        snprintf(a,sizeof(a),"%" PRIu64,va);snprintf(b,sizeof(b),"%" PRIu64,vb);
        /* Only use the prompt, never teacher-force the reference scratchpad. */
        char *input=prompt_text(a,b,0);
        size_t length=input ? cn_encode(input,strlen(input),1,t,(size_t)cap) : SIZE_MAX;
        free(input);
        if(length==SIZE_MAX || length>=(size_t)cap){cn_context_free(s);return 0;}
        prompt=(int)length;
        cn_context_reset(s);const float *x=NULL;
        for(int j=0;j<prompt;j++) if(!cn_forward(s,t[j],&x)){cn_context_free(s);return 0;}
        int n=prompt;
        while(n<cap) {
            int id=best(x);
            if(id<0){cn_context_free(s);return 0;}
            t[n++]=id;
            if(id==CN_EOT) break;
            if(n<cap && !cn_forward(s,id,&x)){cn_context_free(s);return 0;}
        }
        int correct=(int)add_chat_outcome_reward(a,b,cot,t+prompt,n-prompt);
        r->exact.correct+=correct;r->by_width[level]+=correct;
    }
    cn_context_free(s);return 1;
}
int add_chat_validate_format(cn_model *m, int cot, add_chat_validation *result, int fields) {
    if (!m || !result) return 0;
    memset(result, 0, sizeof(*result));
    result->exact.total = 200;
    cn_context *s = cn_context_create(m, 0);
    if (!s)
        return 0;
    uint64_t rng = 20261019;
    const int widths[] = {1, 2, 4, 6, 10};
    for (int i = 0; i < 200; i++) {
        char a[12], b[12];
        int level = i / 40;
        uint64_t power = 1;
        for (int j = 0; j < widths[level]; j++) power *= 10;
        uint64_t va = add_random(&rng) % power, vb = add_random(&rng) % power;
        if (i % 40 == 0) { va = 0; vb = 0; }
        if (i % 40 == 1) { va = power - 1; vb = 1; }
        if (i % 40 == 2) va = vb = power - 1;
        snprintf(a, sizeof(a), "%" PRIu64, va);
        snprintf(b, sizeof(b), "%" PRIu64, vb);
        int t[ADD_CHAT_CAPACITY], prompt,
            n = add_chat_example_format(a, b, cot, 0, t, ADD_CHAT_CAPACITY, &prompt, fields), match = n > 0;
        if (n > cn_model_config(m)->max_seq) { cn_context_free(s); return 0; }
        const float *x = NULL;
        cn_context_reset(s);
        for (int j = 0; match && j < prompt; j++)
            if (!cn_forward(s, t[j], &x))
                match = 0;
        for (int j = prompt; x && j < n; j++) {
            int correct = best(x) == t[j];
            match &= correct;
            result->token_correct += correct; result->token_total++;
            double mx = x[0], sum = 0;
            for (int k = 1; k < CN_VOCAB; k++) if (x[k] > mx) mx = x[k];
            for (int k = 0; k < CN_VOCAB; k++) sum += exp(x[k] - mx);
            result->loss += log(sum) + mx - x[t[j]];
            if (j < n - 1 && !cn_forward(s, t[j], &x)) { cn_context_free(s); return 0; }
        }
        result->exact.correct += match;
        result->by_width[level] += match;
    }
    cn_context_free(s);
    if (!result->token_total) return 0;
    result->loss /= result->token_total;
    result->exact.probability = exp(-result->loss);
    return 1;
}
add_result add_chat_check(cn_model *m, int cot) {
    add_chat_validation result;
    if (!add_chat_validate(m, cot, &result)) return (add_result){0, 200, 0, 0};
    return result.exact;
}
int add_chat_example(const char *a, const char *b, int cot, int copy, int *tokens, int capacity,
                     int *prompt) {
    return add_chat_example_format(a,b,cot,copy,tokens,capacity,prompt,1);
}
int add_chat_predict(cn_model *m, const char *a, const char *b, int cot, const char *json_path,
                     const char *raw_path) {
    return add_chat_predict_format(m,a,b,cot,json_path,raw_path,1);
}
int add_chat_update(cn_model *m, cn_model *ref, int stage, int step, int total, int batch, int cot,
                    uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward,
                    int *informative) {
    return add_chat_update_format(m,ref,stage,step,total,batch,cot,rng,opt,stats,reward,informative,1);
}
int add_chat_validate(cn_model *m, int cot, add_chat_validation *result) {
    return add_chat_validate_format(m,cot,result,1);
}
