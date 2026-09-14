#include "gguf.h"
#include "quant.h"
#include "tokenizer.h"
#include "util.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G4_VOCAB_SIZE 262144
#define G4_ROPE_THETA_SLIDING 10000.0f
#define G4_ROPE_THETA_GLOBAL 1000000.0f
#define G4_FINAL_LOGIT_SOFTCAPPING 30.0f
#define G4_RMS_NORM_EPS 1.0e-6f
#define G4_PARTIAL_ROTARY_FACTOR 0.25f

typedef struct {
  const char *name;
  uint32_t    context_length;
  uint32_t    num_hidden_layers;   // block_count
  uint32_t    hidden_size;         // embedding_length
  uint32_t    intermediate_size;   // feed_forward_length
  uint32_t    num_attention_heads; // head_count
  uint32_t    num_key_value_heads; // head_count_kv
  uint32_t    head_dim;            // key_length_swa + value_length_swa
  uint32_t    sliding_window;
  uint32_t    num_global_key_value_heads;
  uint32_t    global_head_dim;      // key_length
  uint32_t    num_kv_shared_layers; // shared_kv_layers
  uint32_t    use_double_wide_mlp;

  // for the e2b and e4b variant
  uint32_t    hidden_size_per_layer_input; // embedding_length_per_layer_input

  // For the 26B variant
  uint32_t    num_experts;   // expert_count
  uint32_t    top_k_experts; // expert_used_count
  uint32_t    expert_intermediate_size;
} g4_config;

typedef enum {
  G4_E2B             = 0,
  G4_E4B             = 1,
  G4_26B             = 2,
  G4_31B             = 3,
  G4_VARIANT_COUNT   = 4,
  G4_VARIANT_INVALID = -1,
} g4_variant;

static const g4_config G4_CONFIGS[G4_VARIANT_COUNT] = {
  {
    .name                        = "Gemma-4-E2B-It",
    .context_length              = 131072,
    .num_hidden_layers           = 35,
    .hidden_size                 = 1536,
    .intermediate_size           = 6144,
    .num_attention_heads         = 8,
    .num_key_value_heads         = 1,
    .head_dim                    = 256,
    .sliding_window              = 512,
    .num_global_key_value_heads  = 0,
    .global_head_dim             = 512,
    .num_kv_shared_layers        = 20,
    .hidden_size_per_layer_input = 256,
    .use_double_wide_mlp         = 1,
    .num_experts                 = 0,
    .top_k_experts               = 0,
    .expert_intermediate_size    = 0,
  },

  {
    .name                        = "Gemma-4-E4B-It",
    .context_length              = 131072,
    .num_hidden_layers           = 42,
    .hidden_size                 = 2560,
    .intermediate_size           = 10240,
    .num_attention_heads         = 8,
    .num_key_value_heads         = 2,
    .head_dim                    = 256,
    .sliding_window              = 512,
    .num_global_key_value_heads  = 0,
    .global_head_dim             = 512,
    .num_kv_shared_layers        = 18,
    .use_double_wide_mlp         = 0,
    .hidden_size_per_layer_input = 256,
    .num_experts                 = 0,
    .top_k_experts               = 0,
    .expert_intermediate_size    = 0,
  },

  {
    .name                        = "Gemma-4-26B-A4B-It",
    .context_length              = 262144,
    .num_hidden_layers           = 30,
    .hidden_size                 = 2816,
    .intermediate_size           = 2112,
    .num_attention_heads         = 16,
    .num_key_value_heads         = 8,
    .head_dim                    = 256,
    .sliding_window              = 1024,
    .num_global_key_value_heads  = 2,
    .global_head_dim             = 512,
    .num_kv_shared_layers        = 0,
    .use_double_wide_mlp         = 0,
    .hidden_size_per_layer_input = 0,
    .num_experts                 = 128,
    .top_k_experts               = 8,
    .expert_intermediate_size    = 704,
  },

  {
    .name                        = "Gemma-4-31B-It",
    .context_length              = 262144,
    .num_hidden_layers           = 60,
    .hidden_size                 = 3072,
    .intermediate_size           = 21504,
    .num_attention_heads         = 32,
    .num_key_value_heads         = 16,
    .head_dim                    = 256,
    .sliding_window              = 1024,
    .num_global_key_value_heads  = 4,
    .global_head_dim             = 512,
    .num_kv_shared_layers        = 0,
    .use_double_wide_mlp         = 0,
    .hidden_size_per_layer_input = 0,
    .num_experts                 = 0,
    .top_k_experts               = 0,
    .expert_intermediate_size    = 0,
  },
};

typedef struct {
  g4_tensor *attn_norm;
  g4_tensor *attn_q;
  g4_tensor *attn_k;
  g4_tensor *attn_v;
  g4_tensor *attn_out;

  g4_tensor *attn_q_norm;
  g4_tensor *attn_k_norm;
  g4_tensor *attn_post_norm;
  g4_tensor *out_scale;
  g4_tensor *rope_freqs;

  g4_tensor *ffn_norm;
  g4_tensor *ffn_gate;
  g4_tensor *ffn_up;
  g4_tensor *ffn_down;
  g4_tensor *ffn_post_norm;

  g4_tensor *ffn_gate_inp;
  g4_tensor *ffn_gate_inp_s;
  g4_tensor *ffn_pre_norm_2;
  g4_tensor *ffn_post_norm_1;
  g4_tensor *ffn_post_norm_2;

  g4_tensor *ffn_gate_up_exps;
  g4_tensor *ffn_gate_exps;
  g4_tensor *ffn_up_exps;
  g4_tensor *ffn_down_exps;

  g4_tensor *ple_inp_gate;
  g4_tensor *ple_proj;
  g4_tensor *ple_post_norm;
} g4_layer;

typedef struct {
  g4_tensor *token_embd;
  g4_tensor *output;
  g4_tensor *output_norm;

  g4_tensor *per_layer_token_embd;
  g4_tensor *per_layer_model_proj;
  g4_tensor *per_layer_proj_norm;

  g4_layer  *layers;
} g4_weights;

static g4_tensor *blk(const gguf *g, int il, const char *suffix) {
  char name[96];
  snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
  return find_tensor(g, name);
}

static g4_tensor *blk_r(const gguf *g, int il, const char *suffix) {
  char name[96];
  snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
  return required_tensor(g, name);
}

static void bind_layer_weights(const g4_config *cfg, const gguf *g, g4_layer *layer, uint32_t il) {
  layer->attn_norm      = blk_r(g, il, "attn_norm.weight");
  layer->attn_q         = blk_r(g, il, "attn_q.weight");
  layer->attn_out       = blk_r(g, il, "attn_output.weight");
  layer->attn_q_norm    = blk_r(g, il, "attn_q_norm.weight");
  layer->attn_post_norm = blk_r(g, il, "post_attention_norm.weight");
  layer->ffn_norm       = blk_r(g, il, "ffn_norm.weight");
  layer->ffn_gate       = blk_r(g, il, "ffn_gate.weight");
  layer->ffn_up         = blk_r(g, il, "ffn_up.weight");
  layer->ffn_down       = blk_r(g, il, "ffn_down.weight");
  layer->ffn_post_norm  = blk_r(g, il, "post_ffw_norm.weight");
  layer->out_scale      = blk(g, il, "layer_output_scale.weight"); // optional

  // TODO: Conditional require these tensors. Are required only on tensors ownig the kv cache
  // FIXME:
  layer->attn_k      = blk_r(g, il, "attn_k.weight");
  layer->attn_k_norm = blk_r(g, il, "attn_k_norm.weight");
  layer->attn_v      = blk_r(g, il, "attn_v.weight");

  // TODO: rope_freqs is required only in global attn layers only
  // FIXME:
  layer->rope_freqs = required_tensor(g, "rope_freqs.weight");

  // OPTIONAL WEIGHTS ie Variant specific
  // MOE
  layer->ffn_gate_inp = blk(g, il, "ffn_gate_inp.weight");
  if (layer->ffn_gate_inp != NULL) {
    layer->ffn_gate_inp_s   = blk_r(g, il, "ffn_gate_inp.scale");
    layer->ffn_pre_norm_2   = blk_r(g, il, "pre_ffw_norm_2.weight");
    layer->ffn_post_norm_1  = blk_r(g, il, "post_ffw_norm_1.weight");
    layer->ffn_post_norm_2  = blk_r(g, il, "post_ffw_norm_2.weight");
    layer->ffn_gate_up_exps = blk(g, il, "ffn_gate_up_exps.weight");

    if (layer->ffn_gate_up_exps == NULL) {
      layer->ffn_gate_exps    = blk_r(g, il, "ffn_gate_exps.weight");
      layer->ffn_gate_up_exps = blk_r(g, il, "ffn_up_exps.weight");
    }
    layer->ffn_down_exps = blk_r(g, il, "ffn_down_exps.weight");
  }

  // PLE
  if (cfg->hidden_size_per_layer_input > 0) {
    layer->ple_inp_gate  = blk_r(g, il, "inp_gate.weight");
    layer->ple_proj      = blk_r(g, il, "proj.weight");
    layer->ple_post_norm = blk_r(g, il, "post_norm.weight");
  }
}

static void weights_free(g4_weights *weights) {
  if (!weights)
    return;
  free(weights->layers);
  free(weights);
}

static g4_weights *bind_weights(const g4_config *cfg, const gguf *g) {
  g4_weights *weights = xcalloc(1, sizeof(*weights));

  weights->token_embd  = required_tensor(g, "token_embd.weight");
  weights->output      = find_tensor(g, "output.weight");
  weights->output_norm = required_tensor(g, "output_norm.weight");
  if (!weights->output)
    weights->output = weights->token_embd; // tie weights

  if (cfg->hidden_size_per_layer_input > 0) {
    weights->per_layer_token_embd = find_tensor(g, "per_layer_token_embd.weight");
    weights->per_layer_model_proj = find_tensor(g, "per_layer_model_proj.weight");
    weights->per_layer_proj_norm  = find_tensor(g, "per_layer_proj_norm.weight");
  }

  weights->layers = xcalloc(cfg->num_hidden_layers, sizeof(g4_layer));

  for (uint32_t i = 0; i < cfg->num_hidden_layers; i++) {
    bind_layer_weights(cfg, g, &weights->layers[i], i);
  }

  return weights;
}

static g4_variant identify_model(const gguf *g) {
  gguf_str basename;

  if (read_kv_str(g, "general.basename", &basename)) {
    if (streq(&basename, "Gemma-4-E2B-It"))
      return G4_E2B;

    if (streq(&basename, "Gemma-4-E4B-It"))
      return G4_E4B;

    if (streq(&basename, "Gemma-4-26B-A4B-It"))
      return G4_26B;

    if (streq(&basename, "Gemma-4-31B-It"))
      return G4_31B;
  }

  return G4_VARIANT_INVALID;
}

static uint32_t g4_validate_config(const gguf *g, const g4_config *cfg) {
  uint32_t block_count = required_kv_u32(g, "gemma4.block_count");
  uint32_t hidden_size = required_kv_u32(g, "gemma4.embedding_length");

  // TODO: more checks
  if (block_count != cfg->num_hidden_layers) {
    fprintf(stderr, "Error: block count mismatch: GGUF=%u config=%u\n", block_count, cfg->num_hidden_layers);
    return 0;
  }

  if (hidden_size != cfg->hidden_size) {
    fprintf(stderr, "Error: hidden size mismatch: GGUF=%u config=%u\n", hidden_size, cfg->hidden_size);
    return 0;
  }

  return 1;
}

typedef struct {
  gguf            *g;
  const g4_config *cfg;
  g4_weights      *weights;
  g4_tokenizer    *tok;
} g4_ctx;

static void g4_free(g4_ctx *ctx) {
  if (!ctx)
    return;
  tokenizer_free(ctx->tok);
  weights_free(ctx->weights);
  gguf_close(ctx->g);
  free(ctx);
}
static g4_ctx *g4_load(const char *model_dir, const char *prompt) {
  g4_ctx *ctx = xcalloc(1, sizeof(*ctx));

  gguf   *g = gguf_open(model_dir);
  if (!g) {
    fprintf(stderr, "Error: Failed to open GGUF file\n");
    free(ctx);
    return NULL;
  }

  gguf_str arch;
  if (!read_kv_str(g, "general.architecture", &arch) || arch.len == 0) {
    fprintf(stderr, "Error: missing general.architecture\n");
    gguf_close(g);
    free(ctx);
    return NULL;
  }

  g4_variant variant = identify_model(g);

  if (variant == G4_VARIANT_INVALID) {
    fprintf(stderr, "Error: unsupported Gemma 4 model\n");
    gguf_close(g);
    free(ctx);
    return NULL;
  }

  const g4_config *cfg = &G4_CONFIGS[variant];

  if (!g4_validate_config(g, cfg)) {
    gguf_close(g);
    free(ctx);
    return NULL;
  }

  g4_weights   *weights = bind_weights(cfg, g);
  g4_tokenizer *tok     = load_tokenizer(g);

  ctx->cfg     = cfg;
  ctx->g       = g;
  ctx->tok     = tok;
  ctx->weights = weights;

  return ctx;
}

typedef struct {
  uint8_t *buffer;
  size_t   capacity;
  size_t   head;
  size_t   peak_used;
} g4_scratch;

void scratch_init(g4_scratch *alloc, size_t capacity) {
  // TODO: should be conditional on the avaliable architecture ie 32 for avx2 and 64 for avx-512
  capacity      = (capacity + 63) & ~63;
  alloc->buffer = (uint8_t *)aligned_alloc(64, capacity);
  if (!alloc->buffer) {
    fprintf(stderr, "Failed to allocate scratch buffer of %zu bytes\n", capacity);
    exit(1);
  }
  alloc->capacity  = capacity;
  alloc->head      = 0;
  alloc->peak_used = 0;
}

static inline void scratch_reset(g4_scratch *alloc) {
  alloc->head = 0;
}

static inline void *scratch_alloc(g4_scratch *alloc, size_t size) {
  size_t aligned_size = (size + 63) & ~63;
  size_t new_head     = alloc->head + aligned_size;

  if (new_head > alloc->capacity) {
    fprintf(stderr, "Scratch buffer overflow: need %zu, capacity %zu\n", new_head, alloc->capacity);
    return NULL;
  }

  void *ptr   = alloc->buffer + alloc->head;
  alloc->head = new_head;

  if (alloc->head > alloc->peak_used) {
    alloc->peak_used = alloc->head;
  }

  return ptr;
}

static inline void *scratch_calloc(g4_scratch *alloc, size_t size) {
  void *ptr = scratch_alloc(alloc, size);
  if (ptr)
    memset(ptr, 0, size);
  return ptr;
}

// get current high-water mark (for debugging)
size_t scratch_peak_used(g4_scratch *alloc) {
  return alloc->peak_used;
}

void scratch_free(g4_scratch *alloc) {
  free(alloc->buffer);
  alloc->buffer   = NULL;
  alloc->capacity = 0;
  alloc->head     = 0;
}

size_t calculate_max_scratch(g4_ctx *ctx, int T) {
  const g4_config *cfg = ctx->cfg;

  const size_t     H   = cfg->hidden_size;
  const size_t     I   = cfg->intermediate_size;
  const size_t     nh  = cfg->num_attention_heads;
  const size_t     hd  = cfg->head_dim;
  const size_t     E   = cfg->num_experts;
  const size_t     K   = cfg->top_k_experts;
  const size_t     fsz = sizeof(float);

  size_t           attn_out = T * H * fsz;
  size_t           q        = T * nh * hd * fsz;
  size_t           kv       = 2 * T * cfg->num_key_value_heads * hd * fsz;
  size_t           scores   = nh * T * fsz; /* worst case: win_len == T */
  size_t           attn     = attn_out + q + kv + scores;

  size_t           ple_proj = 0;
  if (cfg->hidden_size_per_layer_input > 0) {
    size_t epl = cfg->hidden_size_per_layer_input;
    ple_proj   = 4 * T * epl * cfg->num_hidden_layers * fsz;
  }

  size_t ple = 0;
  if (cfg->hidden_size_per_layer_input > 0) {
    size_t epl = cfg->hidden_size_per_layer_input;
    ple        = T * epl * fsz    /* gate */
                 + T * H * fsz    /* pe_out */
                 + T * epl * fsz; /* proj */
  }

  size_t gate_mult = cfg->use_double_wide_mlp ? 2 : 1;
  size_t ffn       = T * (gate_mult + 1) * I * fsz /* gate+up */
                     + T * H * fsz;                /* out */

  size_t moe = 0;
  if (E > 0) {
    moe = T * E * fsz                   /* router */
          + T * K * (sizeof(int) + fsz) /* routing ids+weights */
          + 3 * T * H * fsz;            /* mlp_norm + mlp_out + moe_out */
  }

  size_t layer = ple_proj + attn + ple + (ffn > moe ? ffn : moe);

  return layer;
}

static inline float f16_to_f32(uint16_t h) {

  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x03ff;
  uint32_t bits;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03ff;
      bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }

  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

static const void *tensor_data(const gguf *g, const g4_tensor *t) {
  return g->data + t->abs_offset;
}

void dequant_row_q8_0(const block_q8_0 *blocks, float *out, int n, int num_blocks, float embd_scale) {
  for (int b = 0; b < num_blocks; b++) {
    const float    scale = f16_to_f32(blocks[b].d);
    const uint64_t bn    = n - (b * 32) < 32 ? n - (b * 32) : 32;
    for (int j = 0; j < bn; j++) {
      out[b * 32 + j] = blocks[b].qs[j] * scale * embd_scale;
    }
  }
}

static void embed(gguf *g, g4_tensor *te, id_buf *tokens, float *out, float embd_scale) {
  if (te->type != GGML_TYPE_Q8_0 || te->ndims != 2) {
    g4_die("expected a 2D Q8_0 token embedding tensor");
  }
  const uint64_t    n          = te->dim[0];
  const uint64_t    num_blocks = (n + 31) / 32;

  const block_q8_0 *table = (const block_q8_0 *)tensor_data(g, te);

  for (uint64_t i = 0; i < tokens->len; i++) {
    int token = tokens->data[i];

    if (token < 0 || (uint64_t)token >= te->dim[1]) {
      g4_die("token id is outside the embedding table");
    }

    const block_q8_0 *row = table + (uint64_t)token * num_blocks;

    dequant_row_q8_0(row, out, n, num_blocks, embd_scale);
  }
}

void g4_forward(g4_ctx *ctx, id_buf *tokens, float *logits, uint32_t pos) {
  const g4_config *cfg     = ctx->cfg;
  const int        seq_len = tokens->len;

  g4_scratch       scratch;
  size_t           max_scratch = calculate_max_scratch(ctx, seq_len);
  scratch_init(&scratch, max_scratch);

  float      *hidden   = xmalloc((size_t)seq_len * cfg->hidden_size * sizeof(float));
  float      *residual = xmalloc((size_t)seq_len * cfg->hidden_size * sizeof(float));
  float      *norm_out = xmalloc((size_t)seq_len * cfg->hidden_size * sizeof(float));

  const float embd_scale = sqrtf((float)cfg->hidden_size);
  embed(ctx->g, ctx->weights->token_embd, tokens, hidden, embd_scale);

  if (cfg->hidden_size_per_layer_input > 0) {
    float *ple_emb = xmalloc((size_t)seq_len * cfg->num_hidden_layers * cfg->hidden_size_per_layer_input);

    float  ple_embd_scale = sqrtf((float)cfg->hidden_size_per_layer_input);
    embed(ctx->g, ctx->weights->per_layer_token_embd, tokens, ple_emb, ple_embd_scale);
  }

  free(residual);
  free(norm_out);
  scratch_free(&scratch);
}

typedef struct {
  const char *model_dir;
  const char *prompt;
} cli_config;

#define MAX_INPUT_LEN 4096
static volatile int g_interrupted = 0;

static void         signal_handler(int sig) {
  (void)sig;
  g_interrupted = 1;
}

static char *g4_generate(g4_ctx *ctx, const char *prompt) {
  id_buf   ids = g4_encode(ctx->tok, prompt, true);

  uint32_t n_past = 0;

  uint32_t n_prompt = (uint32_t)ids.len;
  if (n_prompt > 1) {
    float *logits = xmalloc((size_t)ids.len * G4_VOCAB_SIZE * sizeof(float));
    g4_forward(ctx, &ids, logits, 0);
    printf("forward works\n");
    n_past = n_prompt;

    free(logits);

    n_past++;
  } else {
    n_past = 1;
  }

  char *text = g4_decode(ctx->tok, ids.data, ids.len, true);
  ib_free(&ids);
  return text;
}

static char *g4_chat(g4_ctx *ctx, const char *prompt) {
  // TODO: format the prompt then generate
  return g4_generate(ctx, prompt);
}

static void run_interactive(g4_ctx *ctx) {
  char input[MAX_INPUT_LEN];

  printf("G4 chat. Commands: 'quit'\n");
  while (true) {
    printf("> ");
    fflush(stdout);

    g_interrupted = 0;
    if (!fgets(input, MAX_INPUT_LEN, stdin)) {
      if (g_interrupted) {
        printf("\n");
        continue;
      }
      printf("\n");
      break;
    }

    size_t len = strlen(input);

    // skip trailing newline
    if (len > 0 && input[len - 1] == '\n') {
      input[len - 1] = '\0';
      len--;
    }

    // skip empty input
    if (len == 0)
      continue;

    if (strcmp(input, "quit") == 0)
      break;

    char *response = g4_chat(ctx, input);
    if (!response) {
      fprintf(stderr, "Error: Generation failed\n");
      continue;
    }

    printf("%s\n", response);
    free(response);
  }
}

static void usage() {
  fprintf(stderr, "Gemma4 E2B Inference\n");
  fprintf(stderr, "Options:\n\n");
  fprintf(stderr, "-m, --model <path>     Path to the model directory\n");
  fprintf(stderr, "-p, --prompt <TEXT>    Text prompt to feed to the model for a single shot response\n");
  fprintf(stderr, "-h, --help             Show this help message\n");
}

static void init_cfg(int argc, char *argv[], cli_config *cfg) {
  *cfg = (cli_config){0};

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];

    if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
      if (++i >= argc)
        g4_die("-m requires an argument");
      cfg->model_dir = argv[i];
    } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
      if (++i >= argc)
        g4_die("-p requires an argument");
      cfg->prompt = argv[i];
    } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      usage();
      exit(0);
    } else {
      usage();
      g4_die("unknown option");
    }
  }

  if (!cfg->model_dir) {
    usage();
    g4_die("-m is required");
  }
}

int main(int argc, char *argv[]) {
  cli_config cfg;
  init_cfg(argc, argv, &cfg);

  g4_ctx *ctx = g4_load(cfg.model_dir, cfg.prompt);
  if (!ctx)
    g4_die("failed to load model");

  if (cfg.prompt) {
    char *response = g4_chat(ctx, cfg.prompt);
    if (response) {
      printf("%s\n", response);
      free(response);
    }
  } else {
    run_interactive(ctx);
  }

  g4_free(ctx);
  return 0;
}
