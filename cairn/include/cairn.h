#ifndef CAIRN_H
#define CAIRN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CN_VOCAB 261
#define CN_IM_START 256
#define CN_IM_END 257
#define CN_EOT 258
#define CN_THINK 259
#define CN_THINK_END 260
#define CN_MAX_LAYERS 64
#define CN_MAX_NGRAM 8
#define CN_MAX_HASH_HEADS 16

typedef enum { CN_F32 = 0, CN_FP8 = 1, CN_MXFP4 = 2, CN_KVFP4 = 3 } cn_dtype;
typedef struct {
    cn_dtype type;
    int rows, cols, block, tiled; /* tiled=1: FP8 32x32 weight scales */
    size_t data_bytes, scale_bytes;
    uint8_t *data, *scales;
} cn_quant;
float cn_fp8_decode(uint8_t value);
uint8_t cn_fp8_encode(float value);
float cn_fp4_decode(uint8_t nibble);
uint8_t cn_fp4_encode(float value);
/* q must be zero-initialized, or hold a previous cn_quant_pack result. */
int cn_quant_pack(cn_quant *q, const float *x, int rows, int cols, cn_dtype type, int tiled);
void cn_quant_unpack(const cn_quant *q, float *out);
void cn_quant_free(cn_quant *q);

/* Bytes use IDs 0..255. Special parsing is opt-in; lengths are binary-safe.
   Return required output length; return SIZE_MAX for invalid input. */
size_t cn_encode(const void *bytes, size_t length, int parse_special, int *out, size_t capacity);
size_t cn_decode(const int *tokens, size_t count, int render_special, void *out, size_t capacity);

typedef struct {
    int enabled, max_order, heads, head_dim, buckets, pad_id;
    /* Zero selects reproducible native defaults. Nonzero permits imported layouts.
       Primes must be distinct over ALL layers, ordered by (order,head). */
    uint32_t primes[(CN_MAX_NGRAM - 1) * CN_MAX_HASH_HEADS];
    uint64_t multipliers[CN_MAX_NGRAM];
} cn_engram_config;
typedef struct {
    int kv_source;    /* -1: SWA only; self: Full; earlier layer: Reuse/Reindex */
    int index_source; /* self: index anew; earlier layer: reuse selected positions */
    int compression;  /* group size for owner (encoder=2, decoder=1) */
    cn_engram_config engram;
} cn_layer_config;
typedef struct {
    int vocab, dim, layers, encoder_layers, max_seq, window;
    int heads, head_dim, rope_dim, q_rank, o_rank, o_groups;
    int ff_dim, experts, active_experts, shared_experts;
    int index_heads, index_dim, index_topk, candidate_blocks, candidate_size;
    int hc, sinkhorn_iters;
    float norm_eps, hc_eps, rope_theta, compress_theta, route_scale, swiglu_limit;
    int original_seq; /* 0 disables YaRN; SWA-only layers always use base RoPE */
    float rope_factor, beta_fast, beta_slow;
    float index_loss; /* detached attention-distribution distillation for discrete indexer */
    int quant_cache;  /* FP8 SWA, E2M1/E4M3 global, MXFP4 index q/k */
    int qat;          /* quantized forward, straight-through master-weight gradient */
    uint64_t seed;
    int token_map[CN_VOCAB];
    cn_layer_config layer[CN_MAX_LAYERS];
} cn_config;
typedef struct cn_model cn_model;
typedef struct cn_context cn_context;
typedef struct {
    float learning_rate, beta1, beta2, eps, weight_decay, clip_norm, balance_rate;
} cn_optimizer;
typedef struct {
    float loss, index_loss, grad_norm;
    uint64_t step;
    size_t parameters, arena_bytes;
} cn_metrics;

void cn_config_tiny(cn_config *c);
/* Official 40-layer Full/Reindex/Reuse schedule at CPU-friendly widths. */
void cn_config_v41_topology(cn_config *c);
int cn_config_validate(const cn_config *c, char *error, size_t capacity);
/* Copies/validates the structure; no global mutable model or execution state. */
cn_model *cn_model_create(const cn_config *c, char *error, size_t capacity);
void cn_model_free(cn_model *m);
const cn_config *cn_model_config(const cn_model *m);
size_t cn_parameter_count(const cn_model *m);
cn_context *cn_context_create(cn_model *m, int training);
void cn_context_free(cn_context *s);
void cn_context_reset(cn_context *s); /* sequence boundary, including n-grams */
/* Output logits remain valid until the next call/reset. Strict causal exact prefill. */
int cn_forward(cn_context *s, int token, const float **logits);
int cn_prefill(cn_context *s, const int *tokens, size_t count, float *all_logits);
/* Two-phase CED prefill for the last prompt logit and subsequent decoding.
   bounded_replay=0: exact dependency halo; =1: approximate last-window replay.
   Requires an empty inference context and a nonempty encoder/decoder split. */
int cn_prefill_ced(cn_context *s, const int *tokens, size_t count, int bounded_replay, float *last_logits);
const char *cn_context_error(const cn_context *s);
/* Loss over next-token pairs; one sequence, no gradient crossing reset boundaries.
   cn_loss_backward zeroes gradients and accumulates an averaged gradient. */
int cn_loss_backward(cn_model *m, const int *tokens, size_t count, cn_metrics *metrics);
cn_optimizer cn_optimizer_default(void);
/* Begin a new optimizer phase with the current weights. Clears moments,
   gradients, update counter and router usage; does not change route biases. */
int cn_optimizer_reset(cn_model *m);
int cn_optimizer_step(cn_model *m, const cn_optimizer *opt, cn_metrics *metrics);
int cn_train_step(cn_model *m, const int *tokens, size_t count, const cn_optimizer *opt, cn_metrics *metrics);
/* Contiguous unpadded sequences. Each length includes its final target token.
   Exact token-weighted minibatch average; caches/Engram reset between sequences. */
int cn_train_batch(cn_model *m, const int *tokens, const size_t *lengths, size_t batch,
                   const cn_optimizer *opt, cn_metrics *metrics);
/* Explicit per-target objective. Entry i applies to tokens[i], predicted from
   tokens[0..i-1]; entry 0 is ignored. Coefficients are absolute (no implicit
   normalization). ce may be negative for policy gradients. Reference logits
   are detached, vocab-sized, and required only when kl > 0.
   Loss = ce * -log p(target) + kl * KL(p||reference) - entropy * H(p).
   Zero coefficients mask prompt loss, while preserving gradients through it. */
typedef struct {
    float ce, kl, entropy;
    const float *reference_logits;
} cn_token_objective;
/* accumulate=0 clears gradients/router counts; =1 adds to existing gradients.
   auxiliary_weight scales mean sparse-indexer loss separately. No optimizer
   update: call once per sequence, then cn_optimizer_step once per batch. */
int cn_objective_backward(cn_model *m, const int *tokens, size_t count, const cn_token_objective *objective,
                          int accumulate, float auxiliary_weight, cn_metrics *metrics);
int cn_objective_gradient_check(cn_model *m, const int *tokens, size_t count,
                                const cn_token_objective *objective, float epsilon, float tolerance,
                                float *max_error);
/* Optional packed matrix inference. FP32 masters are retained for training.
   type=CN_F32 drops packed weights. Expert matrices use MXFP4 when type=MXFP4. */
int cn_model_quantize(cn_model *m, cn_dtype type);
/* Begin a QAT phase with matching packed inference weights. CN_F32 disables
   QAT and drops packed weights. Rejects immutable models. */
int cn_model_set_qat(cn_model *m, cn_dtype type);
/* Optional decoded-weight inference cache: preserves quantized values and
   activation rounding, trades RAM for avoiding repeated weight unpacking.
   Quantization/optimizer updates invalidate it; prepare again before use. */
int cn_model_inference_cache(cn_model *m, int enabled);
size_t cn_model_inference_cache_bytes(const cn_model *m);
cn_dtype cn_model_precision(const cn_model *m);
/* Number of scalar weights physically stored in the requested packed format. */
size_t cn_model_quantized_parameters(const cn_model *m, cn_dtype type);
/* Drop optimizer/gradient buffers and quantized matrices' FP32 masters.
   The model instance then becomes immutable and inference-only. */
int cn_model_freeze(cn_model *m);
/* Versioned little-endian checkpoint; optimizer optional; atomic temp+rename. */
int cn_save(const cn_model *m, const char *path, int optimizer);
cn_model *cn_load(const char *path, char *error, size_t capacity);
/* Finite-difference sampled checks across every parameter tensor. */
int cn_gradient_check(cn_model *m, const int *tokens, size_t count, float epsilon, float tolerance,
                      float *max_error);
void cn_set_threads(int threads);
int cn_sample(const float *logits, int vocab, float temperature, int top_k, uint64_t *rng);
#ifdef __cplusplus
}
#endif
#endif
