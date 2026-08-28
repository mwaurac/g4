#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
  gguf_str       key;
  uint32_t       type;
  const uint8_t *raw;
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
  uint64_t n_kv;
  uint64_t n_tensors;
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

static int read_bytes(gguf *g, void *dst, uint64_t n) {
  // sanity check
  if ((n > g->size) || (g->offset > g->size - n)) {
    printf("Error: Seeking error");
    return 0;
  }
  memcpy(dst, (g->data + g->offset), n);
  g->offset += n;
  return 1;
}

static int skip_bytes(gguf *g, uint64_t n) {
  // sanity check
  if ((n > g->size) || (g->offset > g->size - n)) {
    printf("Error: Seeking error");
    return 0;
  }
  g->offset += n;
  return 1;
}

static int read_u32(gguf *g, void *dst) {
  return read_bytes(g, dst, 4);
}

static int read_u64(gguf *g, void *dst) {
  return read_bytes(g, dst, 8);
}

static int read_str(gguf *g, gguf_str *dst) {
  uint64_t len;
  if (!read_u64(g, &len))
    return 0;

  dst->len = len;
  dst->ptr = (const char *)g->data + g->offset;
  g->offset += len;

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

static int skip_value(gguf *g, uint32_t type, uint32_t depth) {
  if (depth > 8) {
    printf("Metadata nesting too deep\n");
    return 0;
  }

  int scalar = scalar_value_size(type);
  if (scalar != 0)
    skip_bytes(g, scalar);
  if (type == GGUF_VALUE_STRING) {
    gguf_str skipped;
    return read_str(g, &skipped);
  }
  if (type == GGUF_VALUE_ARRAY) {
    uint64_t len;
    uint32_t etype;

    if (!read_u64(g, &len))
      return 0;
    if (!read_u32(g, &etype))
      return 0;

    uint64_t item_size = scalar_value_size(etype);
    if (item_size != 0) {
      if (len > UINT64_MAX / item_size) {
        printf("Error: Metadata array is too large\n");
        return 0;
      }
      return skip_bytes(g, item_size * len);
    }

    for (uint64_t i = 0; i < len; i++) {
      if (!skip_value(g, etype, depth + 1))
        return 0;
    }
    return 1;
  }

  printf("Error: Unknown metadata type");
  return 0;
}

static int streq(gguf_str *str, const char *s) {
  int slen = strlen(s);
  return (memcmp(str->ptr, s, slen) == 0 && slen == str->len);
}

static void parse_kv(gguf *g) {
  g->kv = calloc(g->n_kv, sizeof(gguf_kv));
  if (!g->kv) {
    fprintf(stderr, "Error: Failed to allocate memory for KV metadata\n");
    return;
  }

  for (uint64_t i = 0; i < g->n_kv; i++) {
    gguf_kv *kv = &g->kv[i];

    if (!read_str(g, &kv->key))
      return; // read the key
    if (!read_u32(g, &kv->type))
      return;

    if (streq(&kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
      if (!read_u32(g, &g->alignment))
        return;
      if (g->alignment == 0) {
        fprintf(stderr, "Error: alignment must be a power of 2\n");
        g->alignment = 32; // bail out
      }
      continue; // already consumed the value, don't skip_value it too
    }
    kv->raw = g->data + g->offset;
    if (!skip_value(g, kv->type, 0))
      return; // TODO: kill program
  }
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

uint64_t get_alignment_padding(uint64_t alignment, uint64_t offset) {
  return (alignment - (offset % alignment)) % alignment;
}

static void parse_tensors(gguf *g) {
  g->tensors = calloc(g->n_tensors, sizeof(g4_tensor));
  if (!g->tensors) {
    fprintf(stderr, "Error: Failed to allocate memory for tensors");
    return;
  }

  for (uint64_t i = 0; i < g->n_tensors; i++) {
    g4_tensor *tensor = &g->tensors[i];

    if (!read_str(g, &tensor->name))
      return; // TODO:kill program when fails
    if (!read_u32(g, &tensor->ndims))
      return;

    if (tensor->ndims > MAX_DIMS || tensor->ndims == 0) {
      fprintf(stderr, "Tensor has unexpected number of dims");
    }

    tensor->elements = 1;
    for (uint64_t i = 0; i < tensor->ndims; i++) {
      if (!read_u64(g, &tensor->dim[i]))
        return;

      tensor->elements *= tensor->dim[i];
    }

    if (!read_u32(g, &tensor->type))
      return;
    if (!read_u64(g, &tensor->offset))
      return;

    if (!tensor_nbytes(tensor->type, tensor->elements, &tensor->bytes)) {
      fprintf(stderr, "Error: tensor has unsupported GGML type");
    }
  }

  uint64_t padding = get_alignment_padding(g->alignment, g->offset);
  g->data_offset   = g->offset + padding;

  for (uint64_t i = 0; i < g->n_tensors; i++) {
    g4_tensor *tensor = &g->tensors[i];

    if (tensor->offset > UINT64_MAX - g->data_offset) {
      fprintf(stderr, "Error: tensor relative offset overflows the data_offset");
    }

    tensor->abs_offset = g->data_offset + tensor->offset;
  }
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
    return NULL;
  }

  g->fd          = fd;
  g->alignment   = 32;
  g->data        = mapped;
  g->header      = mapped;
  g->offset      = sizeof(gguf_header);
  g->data_offset = 0;
  g->size        = (uint64_t)sb.st_size;

  g->n_kv      = g->header->n_kv;
  g->n_tensors = g->header->n_tensors;

  parse_kv(g);
  parse_tensors(g);

  return g;
}
typedef struct {
  const char *model_dir;
} cli_config;

static void usage() {
  fprintf(stderr, "Gemma4 E2B Inference\n");
  fprintf(stderr, "Options:\n\n");
  fprintf(stderr, "-m, --model <path>     Path to the model directory\n");
  fprintf(stderr, "-h, --help             Show this help message\n");
}

static int init_cfg(int argc, char *argv[], cli_config *cfg) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: -m requires an argument\n");
        return 0;
      }
      cfg->model_dir = argv[i];
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return 0;
    } else {
      fprintf(stderr, "Error: Unknown option\n");
    }
  }
  return 1;
}
int main(int argc, char *argv[]) {
  cli_config cfg;

  if (!init_cfg(argc, argv, &cfg)) {
    usage();
    return 1;
  }
  return 0;
}
