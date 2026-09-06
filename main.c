#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define G4_VOCAB_SIZE 262144
#define G4_ROPE_THETA_SLIDING 10000.0f
#define G4_ROPE_THETA_GLOBAL 1000000.0f
#define G4_FINAL_LOGIT_SOFTCAPPING 30.0f
#define G4_RMS_NORM_EPS 1.0e-6f
#define G4_PARTIAL_ROTARY_FACTOR 0.25f

// TODO: replace most areas with die
static void g4_die(const char *msg) {
  fprintf(stderr, "Error: %s\n", msg);
  exit(1);
}

static void *xcalloc(size_t n, size_t size) {
  void *p = calloc(n, size);
  if (!p)
    g4_die("out of memory");
  return p;
}

enum {
  GGML_TYPE_F32     = 0,
  GGML_TYPE_F16     = 1,
  GGML_TYPE_Q4_0    = 2,
  GGML_TYPE_Q4_1    = 3,
  GGML_TYPE_Q5_0    = 6,
  GGML_TYPE_Q5_1    = 7,
  GGML_TYPE_Q8_0    = 8,
  GGML_TYPE_Q8_1    = 9,
  GGML_TYPE_Q2_K    = 10,
  GGML_TYPE_Q3_K    = 11,
  GGML_TYPE_Q4_K    = 12,
  GGML_TYPE_Q5_K    = 13,
  GGML_TYPE_Q6_K    = 14,
  GGML_TYPE_Q8_K    = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS  = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S   = 19,
  GGML_TYPE_IQ4_NL  = 20,
  GGML_TYPE_IQ3_S   = 21,
  GGML_TYPE_IQ2_S   = 22,
  GGML_TYPE_IQ4_XS  = 23,
  GGML_TYPE_I8      = 24,
  GGML_TYPE_I16     = 25,
  GGML_TYPE_I32     = 26,
  GGML_TYPE_I64     = 27,
  GGML_TYPE_F64     = 28,
  GGML_TYPE_IQ1_M   = 29,
  GGML_TYPE_BF16    = 30,
  GGML_TYPE_COUNT
};
typedef struct {
  const char *name;
  uint32_t    block_elems;
  uint32_t    block_bytes;
} ggml_type_info;

static const ggml_type_info ggml_types[] = {
  [0]  = {"f32", 1, 4},
  [1]  = {"f16", 1, 2},
  [2]  = {"q4_0", 32, 18},
  [3]  = {"q4_1", 32, 20},
  [6]  = {"q5_0", 32, 22},
  [7]  = {"q5_1", 32, 24},
  [8]  = {"q8_0", 32, 34},
  [9]  = {"q8_1", 32, 40},
  [10] = {"q2_k", 256, 84},
  [11] = {"q3_k", 256, 110},
  [12] = {"q4_k", 256, 144},
  [13] = {"q5_k", 256, 176},
  [14] = {"q6_k", 256, 210},
  [15] = {"q8_k", 256, 292},
  [16] = {"iq2_xxs", 256, 66},
  [17] = {"iq2_xs", 256, 74},
  [18] = {"iq3_xxs", 256, 98},
  [19] = {"iq1_s", 256, 110},
  [20] = {"iq4_nl", 256, 50},
  [21] = {"iq3_s", 256, 110},
  [22] = {"iq2_s", 256, 82},
  [23] = {"iq4_xs", 256, 136},
  [24] = {"i8", 1, 1},
  [25] = {"i16", 1, 2},
  [26] = {"i32", 1, 4},
  [27] = {"i64", 1, 8},
  [28] = {"f64", 1, 8},
  [29] = {"iq1_m", 256, 56},
  [30] = {"bf16", 1, 2},
};

static const ggml_type_info *tensor_type(uint32_t type) {
  uint32_t n = sizeof(ggml_types) / sizeof(ggml_types[0]);
  if (type >= n || ggml_types[type].name == NULL)
    return NULL;
  return &ggml_types[type];
}

enum {
  GGUF_VALUE_UINT8   = 0,
  GGUF_VALUE_INT8    = 1,
  GGUF_VALUE_UINT16  = 2,
  GGUF_VALUE_INT16   = 3,
  GGUF_VALUE_UINT32  = 4,
  GGUF_VALUE_INT32   = 5,
  GGUF_VALUE_FLOAT32 = 6,
  GGUF_VALUE_BOOL    = 7,
  GGUF_VALUE_STRING  = 8,
  GGUF_VALUE_ARRAY   = 9,
  GGUF_VALUE_UINT64  = 10,
  GGUF_VALUE_INT64   = 11,
  GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
  uint64_t    len;
  const char *ptr;
} gguf_str;

typedef struct {
  uint32_t etype;
  uint64_t len;
  uint64_t data_pos;
} gguf_array;

typedef struct {
  gguf_str key;
  uint32_t type;
  uint64_t val_pos;
} gguf_kv;

#define MAX_DIMS 4
typedef struct {
  gguf_str name;
  uint32_t ndims;
  uint64_t dim[MAX_DIMS];
  uint32_t type;
  uint64_t offset;
  uint64_t abs_offset;
  uint64_t elements;
  uint64_t bytes;
} g4_tensor;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint64_t n_tensors;
  uint64_t n_kv;
} gguf_header;

typedef struct {
  int            fd;
  const uint8_t *data;
  uint64_t       size;

  uint32_t       alignment;
  uint64_t       offset;
  uint64_t       data_offset;

  uint64_t       n_kv;
  uint64_t       n_tensors;
  gguf_header   *header;
  gguf_kv       *kv;
  g4_tensor     *tensors;
} gguf;

typedef struct {
  const uint8_t *base;
  uint64_t       size;
  uint64_t       offset;
  char           error[256];
} g4_cursor;

static g4_cursor cursor_at(const gguf *g, const uint64_t offset) {
  g4_cursor c = {
    .base   = g->data,
    .size   = g->size,
    .offset = offset,
    .error  = {0},
  };
  return c;
}

static int cursor_fail(g4_cursor *c, const char *msg) {
  if (c->error[0] == '\0')
    snprintf(c->error, sizeof(c->error), "%s", msg);
  return 0;
}

static int cursor_read_bytes(g4_cursor *c, void *dst, uint64_t n) {
  if (n > c->size || c->offset > c->size - n)
    return cursor_fail(c, "read past end of file");
  memcpy(dst, c->base + c->offset, n);
  c->offset += n;
  return 1;
}

static int cursor_skip(g4_cursor *c, uint64_t n) {
  if (n > c->size || c->offset > c->size - n)
    return cursor_fail(c, "seek past end of file");
  c->offset += n;
  return 1;
}

static int cursor_u32(g4_cursor *c, uint32_t *dst) {
  return cursor_read_bytes(c, dst, 4);
}

static int cursor_u64(g4_cursor *c, uint64_t *dst) {
  return cursor_read_bytes(c, dst, 8);
}

static int cursor_str(g4_cursor *c, gguf_str *dst) {
  uint64_t len;
  if (!cursor_u64(c, &len))
    return 0;
  if (len > c->size || c->offset > c->size - len)
    return cursor_fail(c, "string extends past end of file");
  dst->len = len;
  dst->ptr = (const char *)c->base + c->offset;
  c->offset += len;

  return 1;
}

static int scalar_value_size(uint32_t type) {
  switch (type) {
  case GGUF_VALUE_UINT8:
  case GGUF_VALUE_INT8:
  case GGUF_VALUE_BOOL:
    return 1;
  case GGUF_VALUE_UINT16:
  case GGUF_VALUE_INT16:
    return 2;
  case GGUF_VALUE_UINT32:
  case GGUF_VALUE_INT32:
  case GGUF_VALUE_FLOAT32:
    return 4;
  case GGUF_VALUE_UINT64:
  case GGUF_VALUE_INT64:
  case GGUF_VALUE_FLOAT64:
    return 8;
  default:
    return 0;
  }
}

static int skip_value(g4_cursor *c, uint32_t type, uint32_t depth) {
  if (depth > 8)
    return cursor_fail(c, "metadata nesting too deep");

  int scalar = scalar_value_size(type);
  if (scalar != 0)
    return cursor_skip(c, (uint64_t)scalar);
  if (type == GGUF_VALUE_STRING) {
    gguf_str skipped;
    return cursor_str(c, &skipped);
  }
  if (type == GGUF_VALUE_ARRAY) {
    uint64_t len;
    uint32_t etype;

    if (!cursor_u32(c, &etype))
      return 0;
    if (!cursor_u64(c, &len))
      return 0;

    uint64_t item_size = scalar_value_size(etype);
    if (item_size != 0) {
      if (len > UINT64_MAX / item_size)
        return cursor_fail(c, "metadata array is too large");
      return cursor_skip(c, item_size * len);
    }

    for (uint64_t i = 0; i < len; i++) {
      if (!skip_value(c, etype, depth + 1))
        return 0;
    }
    return 1;
  }

  return cursor_fail(c, "unknown metadata type");
}

static int streq(const gguf_str *str, const char *s) {
  uint64_t slen = strlen(s);
  return (memcmp(str->ptr, s, slen) == 0 && slen == str->len);
}

static int gstr_cmp(const gguf_str *a, const gguf_str *b) {
  return a->len == b->len && memcmp(a->ptr, b->ptr, a->len) == 0;
}

static int parse_kv(gguf *g, g4_cursor *c) {
  g->kv = calloc(g->n_kv, sizeof(gguf_kv));
  if (!g->kv)
    return cursor_fail(c, "failed to allocate KV metadata");

  for (uint64_t i = 0; i < g->n_kv; i++) {
    gguf_kv *kv = &g->kv[i];

    if (!cursor_str(c, &kv->key))
      return 0;
    if (!cursor_u32(c, &kv->type))
      return 0;

    if (streq(&kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
      if (!cursor_u32(c, &g->alignment))
        return 0;
      if (g->alignment == 0) {
        fprintf(stderr, "Error: alignment must be a power of 2\n");
        g->alignment = 32; // bail out
      }
      continue; // already consumed the value, don't skip_value it too
    }
    kv->val_pos = c->offset;
    if (!skip_value(c, kv->type, 0))
      return 0;
  }
  return 1;
}

static const char *tensor_type_name(uint32_t type) {
  const ggml_type_info *info = tensor_type(type);
  return info ? info->name : "unknown";
}

static int tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
  const ggml_type_info *info = tensor_type(type);
  if (!info || info->block_elems == 0)
    return 0;
  uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
  if (blocks > UINT64_MAX / info->block_bytes)
    return 0;
  *bytes = blocks * info->block_bytes;
  return 1;
}

static uint64_t get_alignment_padding(uint64_t alignment, uint64_t offset) {
  return (alignment - (offset % alignment)) % alignment;
}

static int parse_tensors(gguf *g, g4_cursor *c) {
  g->tensors = calloc(g->n_tensors, sizeof(g4_tensor));
  if (!g->tensors)
    return cursor_fail(c, "failed to allocate tensors");

  for (uint64_t i = 0; i < g->n_tensors; i++) {
    g4_tensor *tensor = &g->tensors[i];

    if (!cursor_str(c, &tensor->name))
      return 0;
    if (!cursor_u32(c, &tensor->ndims))
      return 0;

    if (tensor->ndims > MAX_DIMS || tensor->ndims == 0)
      return cursor_fail(c, "tensor has unexpected number of dims");

    tensor->elements = 1;
    for (uint32_t d = 0; d < tensor->ndims; d++) {
      if (!cursor_u64(c, &tensor->dim[d]))
        return 0;
      if (tensor->dim[d] != 0 && tensor->elements > UINT64_MAX / tensor->dim[d])
        return cursor_fail(c, "tensor element count overflows");

      tensor->elements *= tensor->dim[d];
    }

    if (!cursor_u32(c, &tensor->type))
      return 0;
    if (!cursor_u64(c, &tensor->offset))
      return 0;

    if (!tensor_nbytes(tensor->type, tensor->elements, &tensor->bytes)) {
      snprintf(c->error, sizeof(c->error), "tensor has unsupported GGML type %s", tensor_type_name(tensor->type));
      return 0;
    }
  }

  uint64_t padding = get_alignment_padding(g->alignment, c->offset);
  g->data_offset   = c->offset + padding;

  for (uint64_t i = 0; i < g->n_tensors; i++) {
    g4_tensor *tensor = &g->tensors[i];

    if (tensor->offset > UINT64_MAX - g->data_offset)
      return cursor_fail(c, "tensor relative offset overflows data_offset");

    tensor->abs_offset = g->data_offset + tensor->offset;
  }
  return 1;
}

static void gguf_close(gguf *g) {
  if (!g)
    return;
  free(g->tensors);
  free(g->kv);
  if (g->data)
    munmap((void *)g->data, g->size);
  if (g->fd >= 0)
    close(g->fd);
  memset(g, 0, sizeof(*g));
  g->fd = -1;
  free(g);
}

static gguf *gguf_open(const char *model_dir) {
  int fd = open(model_dir, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Error: Failed to open file\n");
    return NULL;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return NULL;
  }

  void *mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    fprintf(stderr, "Error: Memory mapping failed\n");
    close(fd);
    return NULL;
  }

  gguf *g = calloc(1, sizeof(*g));
  if (!g) {
    munmap(mapped, sb.st_size);
    close(fd);
    return NULL;
  }

  g->fd          = fd;
  g->alignment   = 32;
  g->data        = mapped;
  g->header      = mapped;
  g->data_offset = 0;
  g->size        = (uint64_t)sb.st_size;

  g->n_kv      = g->header->n_kv;
  g->n_tensors = g->header->n_tensors;

  g4_cursor c = cursor_at(g, sizeof(gguf_header));
  if (!parse_kv(g, &c)) {
    fprintf(stderr, "Error: %s\n", c.error[0] ? c.error : "failed to parse GGUF metadata");
    gguf_close(g);
    return NULL;
  }

  if (!parse_tensors(g, &c)) {
    fprintf(stderr, "Error: %s\n", c.error[0] ? c.error : "failed to parse GGUF tensors");
    gguf_close(g);
    return NULL;
  }

  return g;
}

static gguf_kv *find_kv(const gguf *g, const char *key) {
  for (uint64_t i = 0; i < g->n_kv; i++) {
    if (streq(&g->kv[i].key, key))
      return &g->kv[i];
  }
  return NULL;
}

static g4_tensor *find_tensor(const gguf *g, const char *t_name) {
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    if (streq(&g->tensors[i].name, t_name))
      return &g->tensors[i];
  }
  return NULL;
}

static int read_kv_u32(const gguf *g, const char *key, uint32_t *out) {
  gguf_kv *kv = find_kv(g, key);
  if (!kv || kv->type != GGUF_VALUE_UINT32)
    return 0;
  g4_cursor c = cursor_at(g, kv->val_pos);
  return cursor_u32(&c, out);
}

static int read_kv_str(const gguf *g, const char *key, gguf_str *out) {
  gguf_kv *kv = find_kv(g, key);
  if (!kv || kv->type != GGUF_VALUE_STRING)
    return 0;
  g4_cursor c = cursor_at(g, kv->val_pos);
  return cursor_str(&c, out);
}

static int read_kv_array(const gguf *g, const char *key, gguf_array *out) {
  gguf_kv *kv = find_kv(g, key);
  if (!kv || kv->type != GGUF_VALUE_ARRAY)
    return 0;

  g4_cursor c = cursor_at(g, kv->val_pos);
  if (!cursor_u32(&c, &out->etype))
    return 0;
  if (!cursor_u64(&c, &out->len))
    return 0;

  out->data_pos = c.offset;
  return 1;
}

static uint32_t required_kv_u32(const gguf *g, const char *key) {
  uint32_t v = 0;
  if (!read_kv_u32(g, key, &v)) {
    fprintf(stderr, "Error: missing required GGUF key: %s\n", key);
    exit(1);
  }
  return v;
}

static g4_tensor *required_tensor(const gguf *g, const char *t_name) {
  g4_tensor *t = find_tensor(g, t_name);
  if (!t) {
    fprintf(stderr, "Error: missing required tensor: %s\n", t_name);
    exit(1);
  }
  return t;
}

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

typedef struct {
  gguf_str key;
  int32_t  value;
  bool     used;
} str_i32_entry;

typedef struct {
  str_i32_entry *entries;
  uint64_t       capacity;
  uint64_t       len;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
  uint64_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

static void table_init(str_i32_table *t, uint64_t expected_size) {
  t->capacity = next_pow2(expected_size * 2 + 16);
  t->len      = 0;
  t->entries  = xcalloc((size_t)t->capacity, sizeof(str_i32_entry));
}

static void table_free(str_i32_table *t) {
  free(t->entries);
  memset(t, 0, sizeof(*t));
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
  const uint8_t *p = ptr;
  uint64_t       h = 1469598103934665603ull;
  for (uint64_t i = 0; i < len; i++) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// linear probing
static void table_put(str_i32_table *t, gguf_str key, int32_t value) {
  uint64_t mask = t->capacity - 1;
  uint64_t i    = hash_bytes(key.ptr, key.len) & mask;

  while (t->entries[i].used) {
    if (gstr_cmp(&t->entries[i].key, &key)) {
      t->entries[i].value = value;
      return;
    }
    i = (i + 1) & mask;
  }

  t->entries[i].used  = true;
  t->entries[i].key   = key;
  t->entries[i].value = value;
  t->len++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
  if (t->capacity == 0)
    return false;

  uint64_t mask = t->capacity - 1;
  uint64_t i    = hash_bytes(ptr, len) & mask;

  while (t->entries[i].used) {
    gguf_str key = t->entries[i].key;
    if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
      *value = t->entries[i].value;
      return true;
    }
    i = (i + 1) & mask;
  }
  return false;
}

// followind on the instruction at https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
typedef struct {
  gguf_str     *tokens;
  str_i32_table merge_rank;
  str_i32_table token_to_id;
  uint32_t      vocab_size;

  uint32_t      bos_token_id;
  uint32_t      eos_token_id;
  uint32_t      unk_token_id;

  uint32_t      system_token_id;     // "system"
  uint32_t      user_token_id;       // "user"
  uint32_t      model_token_id;      // "model"
  uint32_t      turn_start_token_id; // <|turn>
  uint32_t      turn_end_token_id;   // <turn|>

  uint32_t      think_token_id;         // <|think|>
  uint32_t      channel_start_token_id; //<|channel>
  uint32_t      channel_end_token_id;   // <channel|>

  uint32_t      tool_start_token_id;          // <|tool>
  uint32_t      tool_end_token_id;            // <tool|>
  uint32_t      tool_call_start_token_id;     // <|tool_call>
  uint32_t      tool_call_end_token_id;       // <tool_call|>
  uint32_t      tool_response_start_token_id; // <|tool_response>
  uint32_t      tool_response_end_token_id;   // <tool_response|>

  uint32_t      string_delimiter_token_id; // <|"|>

  uint32_t      image_start_token_id; //<|image>
  uint32_t      image_end_token_id;   //<image|>
  uint32_t      audio_start_token_id; // <|audio>
  uint32_t      audio_end_token_id;   // <audio|>
  uint32_t      image_placeholer_id;  // <|image|>
  uint32_t      audio_placeholder_id; // <|audio|>
} g4_tokenizer;

static void tokenizer_free(g4_tokenizer *tok) {
  free(tok->tokens);
  table_free(&tok->token_to_id);
  table_free(&tok->merge_rank);
  memset(tok, 0, sizeof(*tok));
}

static int table_lookup(const g4_tokenizer *tok, const char *text) {
  int token = -1;
  if (!table_get(&tok->token_to_id, text, strlen(text), &token)) {
    fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
    exit(1);
  }
  return token;
}

static g4_tokenizer *load_tokenizer(gguf *g) {
  g4_tokenizer *tok = (g4_tokenizer *)xcalloc(1, sizeof(*tok));

  gguf_array    tokens;
  gguf_array    merges;

  if (!read_kv_array(g, "tokenizer.ggml.tokens", &tokens) || tokens.etype != GGUF_VALUE_STRING ||
      tokens.len > INT32_MAX) {
    fprintf(stderr, "Error: GGUF token table is missing or invalid\n");
    exit(1);
  }

  if (!read_kv_array(g, "tokenizer.ggml.merges", &merges) || merges.etype != GGUF_VALUE_STRING) {
    fprintf(stderr, "Error: GGUF merge table is missing or invalid\n");
    exit(1);
  }

  tok->vocab_size = tokens.len;
  tok->tokens     = xcalloc((size_t)tok->vocab_size, sizeof(gguf_str));
  table_init(&tok->token_to_id, tokens.len);

  g4_cursor c = cursor_at(g, tokens.data_pos);
  // TODO: there must be a bettwe way cause this looks inefficient
  for (uint32_t i = 0; i < tok->vocab_size; i++) {
    if (!cursor_str(&c, &tok->tokens[i]))
      g4_die(c.error);
    table_put(&tok->token_to_id, tok->tokens[i], (int32_t)i);
  }

  table_init(&tok->merge_rank, merges.len);
  c = cursor_at(g, merges.data_pos);
  for (uint64_t i = 0; i < merges.len; i++) {
    gguf_str merge;
    if (!cursor_str(&c, &merge))
      g4_die(c.error);
    table_put(&tok->merge_rank, merge, (int32_t)i);
  }

  tok->bos_token_id                 = table_lookup(tok, "<bos>");
  tok->eos_token_id                 = table_lookup(tok, "<eos>");
  tok->unk_token_id                 = table_lookup(tok, "<unk>");
  tok->system_token_id              = table_lookup(tok, "system");
  tok->user_token_id                = table_lookup(tok, "user");
  tok->model_token_id               = table_lookup(tok, "model");
  tok->turn_start_token_id          = table_lookup(tok, "<|turn>");
  tok->turn_end_token_id            = table_lookup(tok, "<turn|>");
  tok->think_token_id               = table_lookup(tok, "<|think|>");
  tok->channel_start_token_id       = table_lookup(tok, "<|channel>");
  tok->channel_end_token_id         = table_lookup(tok, "<channel|>");
  tok->tool_start_token_id          = table_lookup(tok, "<|tool>");
  tok->tool_end_token_id            = table_lookup(tok, "<tool|>");
  tok->tool_call_start_token_id     = table_lookup(tok, "<|tool_call>");
  tok->tool_call_end_token_id       = table_lookup(tok, "<tool_call|>");
  tok->tool_response_start_token_id = table_lookup(tok, "<|tool_response>");
  tok->tool_response_end_token_id   = table_lookup(tok, "<tool_response|>");
  tok->string_delimiter_token_id    = table_lookup(tok, "<|\"|>");
  tok->image_start_token_id         = table_lookup(tok, "<|image>");
  tok->image_end_token_id           = table_lookup(tok, "<image|>");
  tok->audio_start_token_id         = table_lookup(tok, "<|audio>");
  tok->audio_end_token_id           = table_lookup(tok, "<audio|>");
  tok->image_placeholer_id          = table_lookup(tok, "<|image|>");
  tok->audio_placeholder_id         = table_lookup(tok, "<|audio|>");

  return tok;
}

/* A dynamic byte buffer */
typedef struct {
  uint8_t *data;
  uint64_t len;
  uint64_t capacity;
} byte_buf;

static void bb_init(byte_buf *b) {
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

static void bb_reserve(byte_buf *b, size_t extra) {
  if (b->len + extra <= b->capacity)
    return;
  size_t new_cap = b->capacity ? b->capacity * 2 : 64;
  while (new_cap < b->len + extra)
    new_cap *= 2;
  b->data     = (uint8_t *)realloc(b->data, new_cap);
  b->capacity = new_cap;
}

static void bb_push(byte_buf *b, const uint8_t *bytes, size_t n) {
  bb_reserve(b, n);
  memcpy(b->data + b->len, bytes, n);
  b->len += n;
}

static void bb_push_byte(byte_buf *b, uint8_t byte) {
  bb_push(b, &byte, 1);
}

static void bb_free(byte_buf *b) {
  free(b->data);
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

typedef struct {
  int32_t *data;
  uint64_t len;
  uint64_t capacity;
} id_buf;

static void ib_init(id_buf *b) {
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

static void ib_push(id_buf *b, int32_t id) {
  if (b->len == b->capacity) {
    uint64_t new_cap = b->capacity ? b->capacity * 2 : 64;
    b->data          = (int32_t *)realloc(b->data, new_cap * sizeof(int32_t));
    b->capacity      = new_cap;
  }
  b->data[b->len++] = id;
}

static void ib_free(id_buf *b) {
  free(b->data);
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

#define SPACE_MARKER_BYTES "\xE2\x96\x81" /* U+2581 "▁" */
static void normalize(const char *text, byte_buf *out) {
  bb_init(out);
  int starts_with_space = (text[0] == ' ');
  if (!starts_with_space)
    bb_push(out, (const uint8_t *)SPACE_MARKER_BYTES, 3);
  for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
    if (*p == ' ')
      bb_push(out, (const uint8_t *)SPACE_MARKER_BYTES, 3);
    else
      bb_push_byte(out, *p);
  }
}

/*
 * UTF-8 rune (codepoint byte-span) decoding
 */

typedef struct {
  uint32_t start;
  uint32_t len;
} rune;

static uint32_t utf8_rune_len(uint8_t lead) {
  if ((lead & 0x80) == 0x00)
    return 1;
  if ((lead & 0xE0) == 0xC0)
    return 2;
  if ((lead & 0xF0) == 0xE0)
    return 3;
  if ((lead & 0xF8) == 0xF0)
    return 4;
  return 1; // invalid lead byte. treat as a single stray byte
}

static uint32_t decode_runes(const uint8_t *buf, uint32_t len, rune **out) {
  uint32_t cap   = len + 1;
  rune    *runes = (rune *)malloc(cap * sizeof(rune));
  uint32_t count = 0;
  uint32_t i     = 0;

  while (i < len) {
    uint32_t rl = utf8_rune_len(buf[i]);
    if (i + rl > len)
      rl = len - 1; // truncated sequence: take what's left
    runes[count].start = i;
    runes[count].len   = rl;
    count++;
    i += rl;
  }
  *out = runes;
  return count;
}

typedef struct {
  int32_t  vocab_id;
  uint32_t byte_start;
  uint32_t byte_len;
} symbol;

static int bpe_rank(const g4_tokenizer *tok, const uint8_t *base, const symbol *a, const symbol *b) {
  uint64_t len = (uint64_t)a->byte_len + 1 + b->byte_len;
  char     stack[512];
  char    *buf = len <= sizeof(stack) ? stack : malloc((size_t)len);

  memcpy(buf, base + a->byte_start, (size_t)a->byte_len);
  buf[a->byte_len] = ' ';
  memcpy(buf + a->byte_len + 1, base + b->byte_start, (size_t)b->byte_len);

  int rank = -1;
  table_get(&tok->merge_rank, buf, len, &rank);

  if (buf != stack)
    free(buf);
  return rank;
}

static void bpe_merge_word(const g4_tokenizer *tok, const uint8_t *base, symbol *symbols, int *count) {
  for (;;) {
    int best_rank = INT32_MAX;
    int best_idx  = -1;

    for (int i = 0; i + 1 < *count; i++) {
      if (symbols[i].vocab_id < 0 || symbols[i + 1].vocab_id < 0)
        continue;
      int rank = bpe_rank(tok, base, &symbols[i], &symbols[i + 1]);
      if (rank >= 0 && rank < best_rank) {
        best_rank = rank;
        best_idx  = i;
      }
    }

    if (best_idx == -1)
      break; // no mergeable adjacent pair left: done

    uint32_t merged_len = symbols[best_idx].byte_len + symbols[best_idx + 1].byte_len;
    int      result_id;
    if (!table_get(&tok->token_to_id, (const char *)(base + symbols[best_idx].byte_start), merged_len, &result_id)) {
      break;
    }

    symbols[best_idx].vocab_id = result_id;
    symbols[best_idx].byte_len = merged_len;
    memmove(&symbols[best_idx + 1], &symbols[best_idx + 2], (size_t)(*count - best_idx - 2) * sizeof(symbol));
    (*count)--;
  }
}

static id_buf g4_encode(const g4_tokenizer *tok, const char *text, bool add_bos) {
  byte_buf norm;
  normalize(text, &norm);

  rune    *runes;
  uint32_t num_runes = decode_runes(norm.data, (uint32_t)norm.len, &runes);

  id_buf   ids;
  ib_init(&ids);
  if (add_bos)
    ib_push(&ids, (int32_t)tok->bos_token_id);

  uint32_t word_start = 0;
  for (uint32_t i = 0; i <= num_runes; i++) {
    int is_marker =
      (i < num_runes && runes[i].len == 3 && memcmp(norm.data + runes[i].start, SPACE_MARKER_BYTES, 3) == 0);
    int word_len_so_far = (int)(i - word_start);
    int at_end          = (i == num_runes);

    if ((is_marker && word_len_so_far > 0) || at_end) {
      int wcount = (int)(i - word_start);
      if (wcount > 0) {
        symbol *symbols = (symbol *)malloc((size_t)wcount * sizeof(symbol));
        for (int k = 0; k < wcount; k++) {
          rune r = runes[word_start + k];
          int  id;
          symbols[k].vocab_id   = table_get(&tok->token_to_id, (const char *)norm.data + r.start, r.len, &id) ? id : -1;
          symbols[k].byte_start = r.start;
          symbols[k].byte_len   = r.len;
        }

        int scount = wcount;
        bpe_merge_word(tok, norm.data, symbols, &scount);

        for (int k = 0; k < scount; k++) {
          if (symbols[k].vocab_id >= 0) {
            ib_push(&ids, symbols[k].vocab_id);
          } else {
            // byte fallback: format "<0xXX>" and look it up directly
            for (uint32_t b = 0; b < symbols[k].byte_len; b++) {
              uint8_t byte = norm.data[symbols[k].byte_start + b];
              char    tag[8];
              snprintf(tag, sizeof(tag), "<0x%02X>", byte);
              int fb;
              if (table_get(&tok->token_to_id, tag, strlen(tag), &fb))
                ib_push(&ids, fb);
              else
                ib_push(&ids, (int32_t)tok->unk_token_id);
            }
          }
        }
        free(symbols);
      }
      word_start = i;
    }
  }

  free(runes);
  bb_free(&norm);
  return ids;
}

static void print_ids(const g4_tokenizer *tok, const id_buf *ids) {
  printf("[");
  for (uint64_t i = 0; i < ids->len; i++) {
    printf("%s%d", i ? ", " : "", ids->data[i]);
  }
  printf("]\npieces: [");
  for (uint64_t i = 0; i < ids->len; i++) {
    int32_t  id    = ids->data[i];
    gguf_str piece = tok->tokens[id];
    printf("%s\"%.*s\"", i ? ", " : "", (int)piece.len, piece.ptr);
  }
  printf("]\n");
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
  ib_free(&ids);

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
