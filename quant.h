#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>

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

const ggml_type_info *tensor_type(uint32_t type);
const char *tensor_type_name(uint32_t type);
int tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);

#endif
