#include "gguf.h"
#include "tokenizer.h"
#include "quant.h"
#include "util.h"

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
  g4_weights *weights = calloc(1, sizeof(*weights));
  if (!weights) {
    fprintf(stderr, "Error: failed to allocate mem\n");
    exit(1);
  }

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

  weights->layers = calloc(cfg->num_hidden_layers, sizeof(g4_layer));
  if (!weights->layers) {
    fprintf(stderr, "Error: Failed to allocate mem for layers\n");
    exit(1);
  }

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

static int g4_load(const char *model_dir, const char *prompt) {
  gguf *g = gguf_open(model_dir);
  if (!g) {
    fprintf(stderr, "Error: Failed to open GGUF file\n");
    return 0;
  }

  gguf_str arch;
  if (!read_kv_str(g, "general.architecture", &arch) || arch.len == 0) {
    fprintf(stderr, "Error: missing general.architecture\n");
    gguf_close(g);
    return 0;
  }

  g4_variant variant = identify_model(g);

  if (variant == G4_VARIANT_INVALID) {
    fprintf(stderr, "Error: unsupported Gemma 4 model\n");
    gguf_close(g);
    return 0;
  }

  const g4_config *cfg = &G4_CONFIGS[variant];

  if (!g4_validate_config(g, cfg)) {
    fprintf(stderr, "Error: GGUF does not match expected config\n");
    gguf_close(g);
    return 0;
  }

  g4_weights   *weights = bind_weights(cfg, g);
  g4_tokenizer *tok     = load_tokenizer(g);

  id_buf        ids = g4_encode(tok, prompt, true);
  print_ids(tok, &ids);

  char *decoded = g4_decode(tok, ids.data, ids.len, true);
  printf("decoded: %s\n", decoded);
  ib_free(&ids);
  free(decoded);

  tokenizer_free(tok);
  weights_free(weights);
  gguf_close(g);
  return 1;
}

typedef struct {
  const char *model_dir;
  const char *prompt;
} cli_config;

static void usage() {
  fprintf(stderr, "Gemma4 E2B Inference\n");
  fprintf(stderr, "Options:\n\n");
  fprintf(stderr, "-m, --model <path>     Path to the model directory\n");
  fprintf(stderr, "-p, --prompt <TEXT>    Text prompt to feed to the model for a single shot response\n");
  fprintf(stderr, "-h, --help             Show this help message\n");
}

static int init_cfg(int argc, char *argv[], cli_config *cfg) {
  *cfg = (cli_config){0};

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];

    if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: -m requires an argument\n");
        return 0;
      }
      cfg->model_dir = argv[i];
    } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: -p requires an argument\n");
        return 0;
      }
      cfg->prompt = argv[i];
    } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
      usage();
      exit(0);
    } else {
      fprintf(stderr, "Error: Unknown option\n");
      return 0;
    }
  }

  if (!cfg->model_dir) {
    fprintf(stderr, "Error: -m is required\n");
    return 0;
  }
  return 1;
}

int main(int argc, char *argv[]) {
  cli_config cfg;

  if (!init_cfg(argc, argv, &cfg)) {
    usage();
    return 1;
  }

  if (!g4_load(cfg.model_dir, cfg.prompt)) {
    fprintf(stderr, "Error: failed to load model\n");
    return 1;
  }
  return 0;
}
