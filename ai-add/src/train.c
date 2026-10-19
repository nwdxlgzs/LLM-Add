#include "add.h"
#include "add_chat.h"
#include "add_column.h"
#include "add_experiment.h"
#include "add_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int token[8];
} local_example;
typedef struct {
    char a[11], b[11];
    int stage;
} problem;
typedef struct {
    int tokens[10][8], count;
    add_trace trace;
    double reward;
    int exact;
} rollout;
add_training add_default_training(void) {
    add_training c = {200, 3000, 400, 4, UINT64_C(20261018)};
    return c;
}
static int save_model(cn_model *m, const char *dir, const char *name, int optimizer) {
    char path[1200];
    return add_path(path, sizeof(path), dir, name) && cn_save(m, path, optimizer);
}
static int read_data(const char *dir, local_example local[200], problem curriculum[2560]) {
    char path[1200];
    if (!add_path(path, sizeof(path), dir, "local.tsv"))
        return 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    int a, b, c, d, e, count = 0, seen[200] = {0};
    while (fscanf(f, "%d %d %d %d %d", &a, &b, &c, &d, &e) == 5) {
        if (count >= 200 || a < 0 || a > 9 || b < 0 || b > 9 || c < 0 || c > 1 || d < 0 || d > 9 ||
            e < 0 || e > 1 || d + 10 * e != a + b + c || seen[(a * 10 + b) * 2 + c]) {
            fclose(f);
            return 0;
        }
        seen[(a * 10 + b) * 2 + c] = 1;
        add_example(a, b, c, local[count++].token);
    }
    int ok = count == 200 && feof(f);
    if (fclose(f))
        ok = 0;
    if (!ok)
        return 0;
    if (!add_path(path, sizeof(path), dir, "curriculum.tsv") || !(f = fopen(path, "rb")))
        return 0;
    count = 0;
    while (count < 2560 && fscanf(f, "%d %10s %10s", &curriculum[count].stage, curriculum[count].a,
                                  curriculum[count].b) == 3) {
        problem *p = curriculum + count;
        if (p->stage != count / 512 || !add_parse(p->a, NULL) || !add_parse(p->b, NULL)) {
            fclose(f);
            return 0;
        }
        count++;
    }
    ok = count == 2560;
    if (fclose(f))
        ok = 0;
    return ok;
}
static double validation(cn_model *m, const problem *p) {
    cn_context *s = cn_context_create(m, 0);
    if (!s)
        return 0;
    int correct = 0;
    for (int i = 0; i < 40; i++) {
        const problem *item = p + 2048 + (i * 11) % 512;
        add_trace t = {0};
        int exact = 0;
        if (add_predict(s, item->a, item->b, &t, NULL))
            add_reward(item->a, item->b, &t, &exact);
        correct += exact;
    }
    cn_context_free(s);
    return correct / 40.;
}
static add_result log_step(FILE *f, cn_model *m, const problem *p, const char *stage, int step,
                           const cn_metrics *stats, double reward, double start) {
    add_result r = add_check(m, NULL);
    double acc = validation(m, p);
    fprintf(f, "%s,%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.6f\n", stage, step, (double)stats->loss,
            r.correct, (double)r.margin, r.probability, acc, reward, add_now() - start);
    fflush(f);
    printf("%-12s step=%5d loss=%8.5f local=%3d/200 margin=%7.3f p=%7.4f curriculum10=%5.1f%% "
           "reward=%.4f time=%.1fs\n",
           stage, step, (double)stats->loss, r.correct, (double)r.margin, r.probability, acc * 100,
           reward, add_now() - start);
    fflush(stdout);
    return r;
}
static int sft_update(cn_model *m, local_example *examples, const int *indices, int count,
                      int batch, uint64_t *rng, const cn_optimizer *opt, cn_metrics *metrics) {
    memset(metrics, 0, sizeof(*metrics));
    for (int k = 0; k < batch; k++) {
        const int *t = examples[indices[add_random(rng) % (unsigned)count]].token;
        cn_token_objective objective[8] = {{0}};
        for (int i = 5; i < 8; i++)
            objective[i].ce = 1.f / (3 * batch);
        cn_metrics one;
        if (!cn_objective_backward(m, t, 8, objective, k != 0, 1.f / batch, &one))
            return 0;
        metrics->loss += one.loss;
        metrics->index_loss += one.index_loss;
        if (one.arena_bytes > metrics->arena_bytes)
            metrics->arena_bytes = one.arena_bytes;
    }
    return cn_optimizer_step(m, opt, metrics);
}
static int sample_rollout(cn_context *s, const problem *p, uint64_t *rng, rollout *r) {
    memset(r, 0, sizeof(*r));
    int carry = 0, n = (int)strlen(p->a);
    r->trace.length = n;
    char reverse[11];
    int written = 0;
    for (int i = 0; i < n; i++) {
        int a = p->a[n - 1 - i] - '0', b = p->b[n - 1 - i] - '0';
        int *t = r->tokens[r->count++];
        t[0] = CN_IM_START;
        t[1] = '0' + a;
        t[2] = '0' + b;
        t[3] = '0' + carry;
        t[4] = CN_THINK;
        cn_context_reset(s);
        const float *x = NULL;
        for (int j = 0; j < 7; j++) {
            if (!cn_forward(s, t[j], &x))
                return 0;
            if (j >= 4) {
                t[j + 1] = cn_sample(x, CN_VOCAB, 1, 0, rng);
                if (t[j + 1] < 0)
                    return 0;
            }
        }
        if (t[5] < '0' || t[5] > '9' || t[6] < '0' || t[6] > '1' || t[7] != CN_THINK_END)
            return 1;
        r->trace.columns[i] = (add_column){a, b, carry, t[5] - '0', t[6] - '0'};
        reverse[written++] = (char)t[5];
        carry = t[6] - '0';
    }
    if (carry)
        reverse[written++] = (char)('0' + carry);
    while (written > 1 && reverse[written - 1] == '0')
        written--;
    for (int i = 0; i < written; i++)
        r->trace.answer[i] = reverse[written - 1 - i];
    r->trace.answer[written] = 0;
    r->trace.valid = 1;
    r->reward = add_reward(p->a, p->b, &r->trace, &r->exact);
    return 1;
}
static int rl_update(cn_model *m, cn_model *ref, const problem *curriculum, int stage,
                     uint64_t *rng, const cn_optimizer *opt, cn_metrics *stats, double *reward_out,
                     int *informative) {
    const int groups = 1, group_size = 4;
    int sequences = 0;
    double reward = 0;
    cn_context *sampler = cn_context_create(m, 0), *reference = cn_context_create(ref, 0);
    if (!sampler || !reference) {
        cn_context_free(sampler);
        cn_context_free(reference);
        return 0;
    }
    memset(stats, 0, sizeof(*stats));
    int ok = 1;
    *informative = 0;
    for (int group = 0; group < groups && ok; group++) {
        const problem *p = curriculum + stage * 512 + add_random(rng) % 512;
        rollout samples[4];
        double sum = 0;
        for (int i = 0; i < group_size; i++) {
            if (!sample_rollout(sampler, p, rng, &samples[i])) {
                ok = 0;
                break;
            }
            sum += samples[i].reward;
            reward += samples[i].reward / (groups * group_size);
        }
        for (int i = 0; i < group_size && ok; i++) {
            float advantage =
                (float)(samples[i].reward - (sum - samples[i].reward) / (group_size - 1));
            if (fabsf(advantage) > 1e-6f)
                *informative = 1;
            for (int c = 0; c < samples[i].count; c++) {
                int *t = samples[i].tokens[c];
                float ref_logits[7 * CN_VOCAB];
                cn_context_reset(reference);
                if (!cn_prefill(reference, t, 7, ref_logits)) {
                    ok = 0;
                    break;
                }
                cn_token_objective objective[8] = {{0}};
                for (int j = 5; j < 8; j++) {
                    objective[j].ce = advantage / (groups * group_size);
                    objective[j].kl = .001f / (groups * group_size);
                    objective[j].entropy = .00001f / (groups * group_size);
                    objective[j].reference_logits = ref_logits + (j - 1) * CN_VOCAB;
                }
                cn_metrics one;
                if (!cn_objective_backward(m, t, 8, objective, sequences++ != 0, 0, &one)) {
                    ok = 0;
                    break;
                }
                stats->loss += one.loss;
            }
        }
    }
    if (ok)
        ok = cn_optimizer_step(m, opt, stats);
    cn_context_free(sampler);
    cn_context_free(reference);
    *reward_out = reward;
    return ok;
}
int add_train(const char *dir, add_training cfg) {
    int ok = 0;
    char path[1200], error[256];
    FILE *log = NULL, *manifest = NULL;
    local_example examples[200];
    problem *curriculum = calloc(2560, sizeof(problem));
    cn_model *m = NULL, *ref = NULL;
    uint64_t rng = cfg.seed;
    double start = add_now();
    if (!curriculum || !add_mkdir(dir) || !read_data(dir, examples, curriculum))
        goto cleanup;
    cn_config c;
    add_config(&c, cfg.seed);
    m = cn_model_create(&c, error, sizeof(error));
    if (!m) {
        fprintf(stderr, "%s\n", error);
        goto cleanup;
    }
    cn_set_threads(1);
    printf("DSV4.1F-small layers=%d encoder=%d dim=%d parameters=%zu; MoE=%d+%d Engram=m4 hc=%d\n",
           c.layers, c.encoder_layers, c.dim, cn_parameter_count(m), c.experts, c.shared_experts,
           c.hc);
    fflush(stdout);
    if (!add_path(path, sizeof(path), dir, "metrics.csv") || !(log = fopen(path, "wb")))
        goto cleanup;
    fprintf(log, "stage,update,loss,local_correct,min_margin,greedy_joint_probability,curriculum10_"
                 "accuracy,rollout_reward,elapsed_seconds\n");
    cn_metrics stats = {0};
    log_step(log, m, curriculum, "initial", 0, &stats, 0, start);
    if (!save_model(m, dir, "00-initial.cn", 1))
        goto cleanup;
    cn_optimizer opt = cn_optimizer_default();
    opt.learning_rate = .002f;
    opt.weight_decay = .001f;
    opt.balance_rate = .0001f;
    opt.beta2 = .99f;
    /* Self-supervised byte language modeling: reconstruct three random input
       symbols, never use an addition label at this stage. */
    for (int step = 1; step <= cfg.pretrain; step++) {
        int tokens[4 * 8];
        size_t lengths[4];
        for (int k = 0; k < 4; k++) {
            int a = (int)(add_random(&rng) % 10), b = (int)(add_random(&rng) % 10),
                carry = (int)(add_random(&rng) % 2);
            int *t = tokens + 8 * k;
            t[0] = CN_IM_START;
            t[1] = '0' + a;
            t[2] = '0' + b;
            t[3] = '0' + carry;
            t[4] = CN_THINK;
            t[5] = '0' + a;
            t[6] = '0' + b;
            t[7] = CN_THINK_END;
            lengths[k] = 8;
        }
        if (!cn_train_batch(m, tokens, lengths, 4, &opt, &stats))
            goto cleanup;
        if (step % 100 == 0 || step == cfg.pretrain)
            log_step(log, m, curriculum, "pretrain", step, &stats, 0, start);
    }
    if (!save_model(m, dir, "01-pretrain.cn", 1))
        goto cleanup;
    int stage_steps[3] = {0};
    for (int stage = 0; stage < 3; stage++) {
        int indices[400], count = 0;
        for (int i = 0; i < 200; i++) {
            int a = examples[i].token[1] - '0', b = examples[i].token[2] - '0',
                carry = examples[i].token[3] - '0';
            if ((stage == 0 && (carry || a + b >= 10)) || (stage == 1 && carry))
                continue;
            indices[count++] = i;
        }
        int base_count = count;
        char name[32];
        snprintf(name, sizeof(name), "sft-%d", stage);
        opt.learning_rate = stage == 0 ? .003f : stage == 1 ? .0015f : .0005f;
        if (stage == 2)
            opt.balance_rate = 0;
        for (int step = 1; step <= cfg.sft; step++) {
            if (!sft_update(m, examples, indices, count, cfg.batch, &rng, &opt, &stats))
                goto cleanup;
            stage_steps[stage] = step;
            if (step % 200 == 0 || step == cfg.sft) {
                add_result result = log_step(log, m, curriculum, name, step, &stats, 0, start);
                if (stage < 2) {
                    cn_context *s = cn_context_create(m, 0);
                    if (!s)
                        goto cleanup;
                    int correct = 0;
                    for (int i = 0; i < count; i++) {
                        int *t = examples[indices[i]].token;
                        add_column col;
                        correct += add_local(s, t[1] - '0', t[2] - '0', t[3] - '0', &col, NULL,
                                             NULL, NULL) &&
                                   col.digit + 10 * col.cout == t[5] - '0' + 10 * (t[6] - '0');
                    }
                    cn_context_free(s);
                    if (correct == count)
                        break;
                } else {
                    if (result.correct == 200 && result.margin >= .5f)
                        break;
                    /* All 200 states are training data. Replay mistakes while
                       retaining 50% uniform coverage to avoid forgetting. */
                    int mistakes[200], errors = 0;
                    cn_context *s = cn_context_create(m, 0);
                    if (!s)
                        goto cleanup;
                    for (int i = 0; i < base_count; i++) {
                        int *t = examples[indices[i]].token;
                        add_column col;
                        if (!add_local(s, t[1] - '0', t[2] - '0', t[3] - '0', &col, NULL, NULL,
                                       NULL) ||
                            col.digit + 10 * col.cout != t[5] - '0' + 10 * (t[6] - '0'))
                            mistakes[errors++] = indices[i];
                    }
                    cn_context_free(s);
                    count = base_count;
                    if (errors)
                        for (int i = 0; i < base_count; i++)
                            indices[count++] = mistakes[i % errors];
                    if (step >= 1000)
                        opt.learning_rate = .0002f;
                }
            }
        }
        snprintf(name, sizeof(name), "02-sft-%d.cn", stage);
        if (!save_model(m, dir, name, 1))
            goto cleanup;
    }
    add_result best = add_check(m, NULL);
    int selected_rl = 0;
    if (!save_model(m, dir, "03-best.cn", 1) || !add_path(path, sizeof(path), dir, "02-sft-2.cn") ||
        !(ref = cn_load(path, error, sizeof(error))))
        goto cleanup;
    if (!cn_optimizer_reset(m))
        goto cleanup;
    opt.learning_rate = .00003f;
    int informative_updates = 0;
    double average_reward = 0;
    for (int step = 1; step <= cfg.rl; step++) {
        double reward;
        int stage = step <= cfg.rl / 3 ? 1 : step <= 2 * cfg.rl / 3 ? 2 : 4;
        int informative;
        if (!rl_update(m, ref, curriculum, stage, &rng, &opt, &stats, &reward, &informative))
            goto cleanup;
        informative_updates += informative;
        average_reward += reward;
        if (step % 50 == 0 || step == cfg.rl) {
            add_result result =
                log_step(log, m, curriculum, "rl-rloo", step, &stats, reward, start);
            if (result.correct > best.correct ||
                (result.correct == best.correct && result.probability > best.probability)) {
                best = result;
                selected_rl = step;
                if (!save_model(m, dir, "03-best.cn", 1))
                    goto cleanup;
            }
        }
    }
    if (!save_model(m, dir, "03-rl-last.cn", 1))
        goto cleanup;
    cn_model_free(m);
    m = NULL;
    if (!add_path(path, sizeof(path), dir, "03-best.cn") ||
        !(m = cn_load(path, error, sizeof(error))))
        goto cleanup;
    if (!save_model(m, dir, "final.cn", 0))
        goto cleanup;
    if (!add_path(path, sizeof(path), dir, "training.json") || !(manifest = fopen(path, "wb")))
        goto cleanup;
    fprintf(manifest,
            "{\n  \"architecture\":\"DSV4.1F reduced CED/CSA2/MoE/Engram/mHC\",\n  \"layers\":6,\n "
            " \"encoder_layers\":3,\n  \"dim\":24,\n  \"parameters\":%zu,\n  \"seed\":%" PRIu64
            ",\n  \"pretrain_steps\":%d,\n  \"sft_steps\":[%d,%d,%d],\n  \"rl_steps\":%d,\n  "
            "\"selected_rl_step\":%d,\n  \"rl_informative_updates\":%d,\n  "
            "\"rl_average_reward\":%.9g,\n  \"local_correct\":%d,\n  \"min_margin\":%.9g,\n  "
            "\"elapsed_seconds\":%.6f\n}\n",
            cn_parameter_count(m), cfg.seed, cfg.pretrain, stage_steps[0], stage_steps[1],
            stage_steps[2], cfg.rl, selected_rl, informative_updates,
            cfg.rl ? average_reward / cfg.rl : 0, best.correct, (double)best.margin,
            add_now() - start);
    ok = best.correct == 200;
cleanup:
    if (log && fclose(log))
        ok = 0;
    if (manifest && fclose(manifest))
        ok = 0;
    free(curriculum);
    cn_model_free(m);
    cn_model_free(ref);
    if (!ok)
        fprintf(stderr,
                "training failed or checkpoint not yet certified: inspect logs/checkpoints\n");
    return ok;
}

/* The same DSV4.1F network and response objective, now with fake-quantized
   weights/activations and FP32 master gradients. No oracle in inference. */
int add_qat(const char *input, const char *dir, int steps, int batch) {
    int ok = 0, selected = 0, actual = 0;
    char path[1200], error[256];
    local_example examples[200];
    problem *curriculum = calloc(2560, sizeof(problem));
    FILE *log = NULL, *manifest = NULL;
    cn_model *m = NULL;
    uint64_t seed = UINT64_C(20261018), rng = seed;
    double start = add_now();
    if (steps < 1 || batch < 1 || batch > 64 || !curriculum || !add_prepare(dir, seed) ||
        !read_data(dir, examples, curriculum))
        goto cleanup;
    m = cn_load(input, error, sizeof(error));
    if (!m) {
        fprintf(stderr, "%s\n", error);
        goto cleanup;
    }
    cn_set_threads(1);
    if (!cn_model_set_qat(m, CN_MXFP4) || !cn_optimizer_reset(m) ||
        !add_path(path, sizeof(path), dir, "qat-metrics.csv") || !(log = fopen(path, "wb")))
        goto cleanup;
    fprintf(log, "stage,update,loss,local_correct,min_margin,greedy_joint_probability,curriculum10_"
                 "accuracy,rollout_reward,elapsed_seconds\n");
    cn_metrics stats = {0};
    add_result initial = log_step(log, m, curriculum, "fp4-ptq", 0, &stats, 0, start);
    add_result best = initial;
    if (!save_model(m, dir, "04-qat-best.cn", 1))
        goto cleanup;
    cn_optimizer opt = cn_optimizer_default();
    opt.learning_rate = .00005f;
    opt.weight_decay = 0;
    opt.balance_rate = 0;
    opt.beta2 = .99f;
    int indices[400], count = 200;
    for (int i = 0; i < 200; i++)
        indices[i] = i;
    for (int step = 1; step <= steps; step++) {
        if (step == 1 || (step - 1) % 25 == 0) {
            /* 50% uniform, 50% incorrect or low-margin states. Includes
               correctly classified but fragile tokens in hard mining. */
            int hard[200], n = 0;
            cn_context *s = cn_context_create(m, 0);
            if (!s)
                goto cleanup;
            for (int i = 0; i < 200; i++) {
                const int *t = examples[i].token;
                add_column col;
                float margin = 0;
                if (!add_local(s, t[1] - '0', t[2] - '0', t[3] - '0', &col, NULL, &margin, NULL) ||
                    col.digit != t[5] - '0' || col.cout != t[6] - '0' || margin < 2.f)
                    hard[n++] = i;
            }
            cn_context_free(s);
            count = 200;
            if (n)
                for (int i = 0; i < 200; i++)
                    indices[count++] = hard[i % n];
        }
        if (step > steps / 2)
            opt.learning_rate = .00002f;
        if (!sft_update(m, examples, indices, count, batch, &rng, &opt, &stats))
            goto cleanup;
        actual = step;
        if (step % 25 == 0 || step == steps) {
            add_result result = log_step(log, m, curriculum, "fp4-qat", step, &stats, 0, start);
            if (result.correct > best.correct ||
                (result.correct == best.correct && result.margin > best.margin)) {
                best = result;
                selected = step;
                if (!save_model(m, dir, "04-qat-best.cn", 1))
                    goto cleanup;
            }
            if (best.correct == 200 && best.margin >= 2.f)
                break;
        }
    }
    if (!save_model(m, dir, "04-qat-last.cn", 1))
        goto cleanup;
    cn_model_free(m);
    m = NULL;
    if (!add_path(path, sizeof(path), dir, "04-qat-best.cn") ||
        !(m = cn_load(path, error, sizeof(error))) || !save_model(m, dir, "final-fp4.cn", 0))
        goto cleanup;
    cn_model_free(m);
    m = NULL;
    /* Validate the file's actual packed bytes, without a new quantization. */
    char checkpoint[1200];
    if (!add_path(checkpoint, sizeof(checkpoint), dir, "final-fp4.cn") ||
        !(m = cn_load(checkpoint, error, sizeof(error))) || cn_model_precision(m) != CN_MXFP4 ||
        !cn_model_quantized_parameters(m, CN_MXFP4) ||
        !add_path(path, sizeof(path), dir, "certificate-fp4-file.json"))
        goto cleanup;
    int certified = add_certify(m, checkpoint, path, 1);
    if (!add_path(path, sizeof(path), dir, "qat-training.json") || !(manifest = fopen(path, "wb")))
        goto cleanup;
    fprintf(manifest,
            "{\n  \"seed\":%" PRIu64
            ",\n  \"steps\":%d,\n  \"selected_step\":%d,\n  \"batch\":%d,\n"
            "  \"initial_correct\":%d,\n  \"local_correct\":%d,\n  \"min_margin\":%.9g,\n"
            "  \"fp4_parameters\":%zu,\n  \"fp8_parameters\":%zu,\n"
            "  \"all_available_backends_certified\":%s,\n  \"elapsed_seconds\":%.6f\n}\n",
            seed, actual, selected, batch, initial.correct, best.correct, (double)best.margin,
            cn_model_quantized_parameters(m, CN_MXFP4), cn_model_quantized_parameters(m, CN_FP8),
            certified ? "true" : "false", add_now() - start);
    ok = certified;
cleanup:
    if (log && fclose(log))
        ok = 0;
    if (manifest && fclose(manifest))
        ok = 0;
    cn_model_free(m);
    free(curriculum);
    if (!ok)
        fprintf(stderr,
                "FP4 QAT did not produce an all-backend certified artifact; inspect logs\n");
    return ok;
}

/* Incremental UI driver. Uses the exact same SFT and on-policy RL updates as
   the command line; yields between optimizer updates for pause/resume. */
struct add_session {
    cn_model *m, *ref;
    local_example examples[200];
    problem curriculum[2560];
    cn_optimizer opt;
    add_progress progress;
    add_result best;
    uint64_t seed, rng;
    char dir[900];
    int indices[400], count, base_count, batch, cot, chat;
    double start;
    float initial_lr;
    add_chat_rl_trace trace;
};
const add_chat_rl_trace *add_session_rollouts(const add_session *s) { return s ? &s->trace : NULL; }
void add_session_free(add_session *s) {
    if (s) {
        cn_model_free(s->m);
        cn_model_free(s->ref);
        free(s);
    }
}
add_session *add_session_create(const cn_config *config, const char *dir) {
    if (!config || !dir || strlen(dir) >= 900)
        return NULL;
    add_session *s = calloc(1, sizeof(*s));
    char error[256];
    if (!s)
        return NULL;
    strcpy(s->dir, dir);
    s->seed = s->rng = config->seed;
    s->cot = 1;
    s->progress.metric_step = -1;
    s->m = cn_model_create(config, error, sizeof(error));
    if (!s->m || !add_prepare(dir, config->seed) || !read_data(dir, s->examples, s->curriculum)) {
        add_session_free(s);
        return NULL;
    }
    return s;
}
cn_model *add_session_model(add_session *s) { return s ? s->m : NULL; }
int add_session_cot(const add_session *s) { return s ? s->cot : 1; }
int add_session_chat(const add_session *s) { return s ? s->chat : 0; }
int add_session_set_chat(add_session *s, int enabled) {
    if (!s || s->progress.running || enabled < 0 || enabled > 3 ||
        (enabled && cn_model_config(s->m)->max_seq <
                        (enabled == 3 ? ADD_CHAT_CAPACITY : ADD_COMPACT_CHAT_CAPACITY)))
        return 0;
    s->chat = enabled;
    return 1;
}
int add_session_set_cot(add_session *s, int enabled) {
    if (!s || s->progress.running || (enabled != 0 && enabled != 1) ||
        (!enabled && cn_model_config(s->m)->max_seq < 40))
        return 0;
    s->cot = enabled;
    return 1;
}
const add_progress *add_session_progress(const add_session *s) { return s ? &s->progress : NULL; }
int add_session_load(add_session *s, const char *path) {
    if (!s || s->progress.running)
        return 0;
    char error[256];
    cn_model *m = cn_load(path, error, sizeof(error));
    if (!m)
        return 0;
    cn_model_free(s->m);
    cn_model_free(s->ref);
    s->m = m;
    s->ref = NULL;
    memset(&s->progress, 0, sizeof(s->progress));
    s->progress.metric_step = -1;
    return 1;
}
add_result add_session_check(add_session *s) {
    add_result r;
    if (s->chat) {
        add_chat_validation validation;
        if (!(s->progress.stage == 7 ? add_chat_validate_outcome(s->m,s->cot,&validation)
              : s->chat == 2 && s->cot ? add_column_validate(s->m, &validation)
                                   : add_chat_validate_format(s->m, s->cot, &validation, s->chat == 3)))
            return (add_result){0,200,0,0};
        r = validation.exact;
        s->progress.validation_loss = validation.loss;
        s->progress.token_correct = validation.token_correct;
        s->progress.token_total = validation.token_total;
        memcpy(s->progress.by_width, validation.by_width, sizeof(validation.by_width));
    } else r = s->cot ? add_check(s->m, NULL) : add_direct_check(s->m);
    s->progress.correct = r.correct;
    s->progress.margin = r.margin;
    s->progress.probability = r.probability;
    s->progress.metric_step = s->progress.step;
    return r;
}
static int session_mine(add_session *s, int qat) {
    int hard[200], n = 0;
    cn_context *context = cn_context_create(s->m, 0);
    if (!context)
        return 0;
    for (int i = 0; i < s->base_count; i++) {
        int id = s->indices[i], *t = s->examples[id].token;
        add_column col;
        float margin = 0;
        if (!add_local(context, t[1] - '0', t[2] - '0', t[3] - '0', &col, NULL, &margin, NULL) ||
            col.digit != t[5] - '0' || col.cout != t[6] - '0' || (qat && margin < 2))
            hard[n++] = id;
    }
    cn_context_free(context);
    s->count = s->base_count;
    if (n)
        for (int i = 0; i < s->base_count; i++)
            s->indices[s->count++] = hard[i % n];
    return 1;
}
int add_session_begin(add_session *s, int stage, int steps, int batch, float lr) {
    if (!s || s->progress.running || stage < 1 || stage > 7 || (stage == 7 && s->chat != 3) || steps < 1 || steps > 100000 ||
        batch < 1 || batch > 32 || !isfinite(lr) || lr < 0 || lr > .01f)
        return 0;
    if(cn_parameter_count(s->m)>ADD_PARAMETER_LIMIT)return 0;
    cn_model_free(s->ref);
    s->ref = NULL;
    s->opt = cn_optimizer_default();
    const float rates[] = {0, .002f, .003f, .0015f, .0005f, .00003f, .00005f, .00002f};
    const float chat_rates[] = {0, .002f, .0015f, .001f, .0007f, .00002f, .00005f, .00002f};
    s->initial_lr = s->opt.learning_rate = lr > 0 ? lr : (s->chat == 1 || s->chat == 3 || !s->cot ? chat_rates : rates)[stage];
    s->opt.beta2 = .99f;
    s->opt.weight_decay = stage >= 6 ? 0 : .001f;
    s->opt.balance_rate = stage < 4 ? .0001f : 0;
    s->batch = batch;
    memset(&s->progress, 0, sizeof(s->progress));
    memset(&s->trace,0,sizeof(s->trace));
    s->progress.stage = stage;
    s->progress.total = steps;
    s->progress.metric_step = -1;
    s->start = add_now();
    /* Imported packed checkpoints may be inference-quantized without QAT.
       Keep their precision and make the policy-gradient forward match the
       sampled policy, with STE masters for the update. */
    if(stage==7 && cn_model_precision(s->m)!=CN_F32 && !cn_model_config(s->m)->qat &&
       !cn_model_set_qat(s->m,cn_model_precision(s->m)))return 0;
    /* Compare QAT against the actual incoming FP32 policy too. If conversion
       makes validation worse, retain that policy and preserve FP4 as an experiment. */
    if (s->chat) {
        s->best = add_session_check(s);
        if (!save_model(s->m, s->dir, "best.cn", 1)) return 0;
        char entry_name[40];
        snprintf(entry_name, sizeof(entry_name), "stage-%d-entry.cn", stage);
        if (!save_model(s->m, s->dir, entry_name, 1)) return 0;
    }
    if (stage == 6) {
        s->rng = s->seed;
        if (!cn_model_set_qat(s->m, CN_MXFP4) || !cn_optimizer_reset(s->m))
            return 0;
    } else if (stage != 7 && cn_model_precision(s->m) != CN_F32 && !cn_model_set_qat(s->m, CN_F32))
        return 0;
    if (stage == 5 || stage == 7) {
        char path[1200], error[256];
        if (!save_model(s->m, s->dir, "reference.cn", 0) ||
            !add_path(path, sizeof(path), s->dir, "reference.cn") ||
            !(s->ref = cn_load(path, error, sizeof(error))) || !cn_optimizer_reset(s->m))
            return 0;
    }
    s->count = 0;
    for (int i = 0; i < 200; i++) {
        const int *t = s->examples[i].token;
        if (stage == 2 && (t[3] != '0' || t[1] + t[2] - 2 * '0' >= 10))
            continue;
        if (stage == 3 && t[3] != '0')
            continue;
        s->indices[s->count++] = i;
    }
    s->base_count = s->count;
    if (!s->chat) s->best = add_session_check(s);
    if (!s->chat && (stage == 5 || stage == 6) && !save_model(s->m, s->dir, "best.cn", 1))
        return 0;
    s->progress.running = 1;
    return 1;
}
int add_session_continue_curriculum(add_session *s, int completed) {
    if(!s || !s->progress.running || s->progress.stage!=4 || s->progress.step ||
       completed<1 || completed>=s->progress.total || s->chat!=3)return 0;
    s->progress.step=s->progress.metric_step=s->progress.best_step=completed;
    return 1;
}
int add_session_tick(add_session *s) {
    if (!s || !s->progress.running)
        return -1;
    add_progress *p = &s->progress;
    int step = p->step + 1, ok = 1, informative = 0;
    cn_metrics stats = {0};
    double reward = 0;
    if (s->chat) {
        if (p->stage != 5 && p->stage != 7) {
            float fraction = (float)step / p->total;
            s->opt.learning_rate = s->initial_lr * (1.f - .8f * fraction);
        }
        ok = s->chat == 2 && s->cot
                 ? add_column_update(s->m, s->ref, p->stage, step, p->total, s->batch, &s->rng,
                                     &s->opt, &stats, &reward, &informative)
                 : add_chat_update_traced(s->m, s->ref, p->stage, step, p->total, s->batch, s->cot, &s->rng,
                                          &s->opt, &stats, &reward, &informative, s->chat == 3, &s->trace);
    } else if (p->stage == 1) {
        int tokens[32];
        size_t lengths[4] = {8, 8, 8, 8};
        for (int k = 0; k < 4; k++) {
            int *t = tokens + k * 8;
            t[0] = CN_IM_START;
            t[1] = '0' + add_random(&s->rng) % 10;
            t[2] = '0' + add_random(&s->rng) % 10;
            t[3] = '0' + add_random(&s->rng) % 2;
            t[4] = CN_THINK;
            t[5] = t[1];
            t[6] = t[2];
            t[7] = CN_THINK_END;
        }
        ok = cn_train_batch(s->m, tokens, lengths, 4, &s->opt, &stats);
    } else if (!s->cot) {
        ok = add_direct_update(s->m, s->ref, p->stage, step, p->total, s->batch, &s->rng, &s->opt,
                               &stats, &reward, &informative);
    } else if (p->stage == 5) {
        int curriculum_stage = step <= p->total / 3 ? 1 : step <= 2 * p->total / 3 ? 2 : 4;
        ok = rl_update(s->m, s->ref, s->curriculum, curriculum_stage, &s->rng, &s->opt, &stats,
                       &reward, &informative);
    } else {
        if (p->stage == 6 && (step == 1 || (step - 1) % 25 == 0))
            ok = session_mine(s, 1);
        if (p->stage == 6 && step > p->total / 2)
            s->opt.learning_rate = s->initial_lr * .4f;
        if (ok)
            ok = sft_update(s->m, s->examples, s->indices, s->count, s->batch, &s->rng, &s->opt,
                            &stats);
    }
    if (!ok) {
        p->running = 0;
        return -1;
    }
    p->step = step;
    p->loss = stats.loss;
    p->arena_bytes = stats.arena_bytes;
    p->reward = reward;
    p->informative += informative;
    if (stats.step) p->optimizer_step = stats.step;
    int interval = p->stage == 1 ? 100 : p->stage <= 4 ? 200 : p->stage == 5 ? 50 : 25;
    if (s->chat) interval = p->stage == 1 ? 100 : 200;
    int done = step >= p->total;
    if (step % interval == 0 || done) {
        add_result r = add_session_check(s);
        if (s->chat) {
            if (p->stage == 1 || r.correct > s->best.correct || (r.correct == s->best.correct && r.probability > s->best.probability)) {
                s->best = r; p->best_step = step;
                if (!save_model(s->m, s->dir, "best.cn", 1)) { p->running = 0; return -1; }
            }
        } else if (!s->cot) {
            /* Direct-answer experiments keep failed updates and use the full
               requested budget, rather than CoT transition early stopping. */
        } else if (p->stage == 2 || p->stage == 3) {
            cn_context *ctx = cn_context_create(s->m, 0);
            if (!ctx) {
                p->running = 0;
                return -1;
            }
            int correct = 0;
            for (int i = 0; i < s->base_count; i++) {
                int *t = s->examples[s->indices[i]].token;
                add_column col;
                correct +=
                    add_local(ctx, t[1] - '0', t[2] - '0', t[3] - '0', &col, NULL, NULL, NULL) &&
                    col.digit == t[5] - '0' && col.cout == t[6] - '0';
            }
            cn_context_free(ctx);
            done |= correct == s->base_count;
        } else if (p->stage == 4) {
            done |= r.correct == 200 && r.margin >= .5f;
            if (!done && !session_mine(s, 0)) {
                p->running = 0;
                return -1;
            }
            if (step >= 1000)
                s->opt.learning_rate = s->initial_lr * .4f;
        } else if (p->stage >= 5) {
            if (r.correct > s->best.correct || (r.correct == s->best.correct &&
                                                (p->stage == 5 ? r.probability > s->best.probability
                                                               : r.margin > s->best.margin))) {
                s->best = r;
                if (!save_model(s->m, s->dir, "best.cn", 1)) {
                    p->running = 0;
                    return -1;
                }
            }
            if (p->stage == 6)
                done |= s->best.correct == 200 && s->best.margin >= 2;
        }
    }
    if (done) {
        p->last_correct = p->correct;
        char last_name[40];
        snprintf(last_name, sizeof(last_name), "stage-%d-last.cn", p->stage);
        if (!save_model(s->m, s->dir, last_name, 1)) {
            p->running = 0;
            return -1;
        }
        if (s->chat || (s->cot && p->stage >= 5)) {
            char path[1200], error[256];
            cn_model *best = NULL;
            if (!add_path(path, sizeof(path), s->dir, "best.cn") ||
                !(best = cn_load(path, error, sizeof(error)))) {
                p->running = 0;
                return -1;
            }
            cn_model_free(s->m);
            s->m = best;
            add_session_check(s);
        }
        char name[40];
        snprintf(name, sizeof(name), "stage-%d.cn", p->stage);
        if (!save_model(s->m, s->dir, name, 1)) {
            p->running = 0;
            return -1;
        }
        p->running = 0;
    }
    p->seconds = add_now() - s->start;
    return done;
}
