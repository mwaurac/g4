#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
  uint64_t elements;
  uint64_t bytes;
  uint8_t *data;
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

  gguf_header    header;
  gguf_kv       *kv;
  g4_tensor     *tensors;
} gguf;

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
