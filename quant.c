#include "quant.h"
#include <stddef.h>

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

const ggml_type_info *tensor_type(uint32_t type) {
  uint32_t n = sizeof(ggml_types) / sizeof(ggml_types[0]);
  if (type >= n || ggml_types[type].name == NULL)
    return NULL;
  return &ggml_types[type];
}

const char *tensor_type_name(uint32_t type) {
  const ggml_type_info *info = tensor_type(type);
  return info ? info->name : "unknown";
}

int tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
  const ggml_type_info *info = tensor_type(type);
  if (!info || info->block_elems == 0)
    return 0;
  uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
  if (blocks > UINT64_MAX / info->block_bytes)
    return 0;
  *bytes = blocks * info->block_bytes;
  return 1;
}
