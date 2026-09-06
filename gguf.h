#ifndef GGUF_H
#define GGUF_H

#include <stdbool.h>
#include <stdint.h>

#define MAX_DIMS 4

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

g4_cursor cursor_at(const gguf *g, uint64_t offset);
int cursor_fail(g4_cursor *c, const char *msg);
int cursor_read_bytes(g4_cursor *c, void *dst, uint64_t n);
int cursor_skip(g4_cursor *c, uint64_t n);
int cursor_u32(g4_cursor *c, uint32_t *dst);
int cursor_u64(g4_cursor *c, uint64_t *dst);
int cursor_str(g4_cursor *c, gguf_str *dst);

int streq(const gguf_str *str, const char *s);
int gstr_cmp(const gguf_str *a, const gguf_str *b);

gguf *gguf_open(const char *model_dir);
void gguf_close(gguf *g);

gguf_kv *find_kv(const gguf *g, const char *key);
g4_tensor *find_tensor(const gguf *g, const char *t_name);
int read_kv_u32(const gguf *g, const char *key, uint32_t *out);
int read_kv_str(const gguf *g, const char *key, gguf_str *out);
int read_kv_array(const gguf *g, const char *key, gguf_array *out);
uint32_t required_kv_u32(const gguf *g, const char *key);
g4_tensor *required_tensor(const gguf *g, const char *t_name);

#endif
