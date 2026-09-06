#include "gguf.h"
#include "quant.h"
#include "util.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

g4_cursor cursor_at(const gguf *g, uint64_t offset) {
  g4_cursor c = {
    .base   = g->data,
    .size   = g->size,
    .offset = offset,
    .error  = {0},
  };
  return c;
}

int cursor_fail(g4_cursor *c, const char *msg) {
  if (c->error[0] == '\0')
    snprintf(c->error, sizeof(c->error), "%s", msg);
  return 0;
}

int cursor_read_bytes(g4_cursor *c, void *dst, uint64_t n) {
  if (n > c->size || c->offset > c->size - n)
    return cursor_fail(c, "read past end of file");
  memcpy(dst, c->base + c->offset, n);
  c->offset += n;
  return 1;
}

int cursor_skip(g4_cursor *c, uint64_t n) {
  if (n > c->size || c->offset > c->size - n)
    return cursor_fail(c, "seek past end of file");
  c->offset += n;
  return 1;
}

int cursor_u32(g4_cursor *c, uint32_t *dst) {
  return cursor_read_bytes(c, dst, 4);
}

int cursor_u64(g4_cursor *c, uint64_t *dst) {
  return cursor_read_bytes(c, dst, 8);
}

int cursor_str(g4_cursor *c, gguf_str *dst) {
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

int streq(const gguf_str *str, const char *s) {
  uint64_t slen = strlen(s);
  return (memcmp(str->ptr, s, slen) == 0 && slen == str->len);
}

int gstr_cmp(const gguf_str *a, const gguf_str *b) {
  return a->len == b->len && memcmp(a->ptr, b->ptr, a->len) == 0;
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

void gguf_close(gguf *g) {
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

gguf *gguf_open(const char *model_dir) {
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

gguf_kv *find_kv(const gguf *g, const char *key) {
  for (uint64_t i = 0; i < g->n_kv; i++) {
    if (streq(&g->kv[i].key, key))
      return &g->kv[i];
  }
  return NULL;
}

g4_tensor *find_tensor(const gguf *g, const char *t_name) {
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    if (streq(&g->tensors[i].name, t_name))
      return &g->tensors[i];
  }
  return NULL;
}

int read_kv_u32(const gguf *g, const char *key, uint32_t *out) {
  gguf_kv *kv = find_kv(g, key);
  if (!kv || kv->type != GGUF_VALUE_UINT32)
    return 0;
  g4_cursor c = cursor_at(g, kv->val_pos);
  return cursor_u32(&c, out);
}

int read_kv_str(const gguf *g, const char *key, gguf_str *out) {
  gguf_kv *kv = find_kv(g, key);
  if (!kv || kv->type != GGUF_VALUE_STRING)
    return 0;
  g4_cursor c = cursor_at(g, kv->val_pos);
  return cursor_str(&c, out);
}

int read_kv_array(const gguf *g, const char *key, gguf_array *out) {
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

uint32_t required_kv_u32(const gguf *g, const char *key) {
  uint32_t v = 0;
  if (!read_kv_u32(g, key, &v)) {
    fprintf(stderr, "Error: missing required GGUF key: %s\n", key);
    exit(1);
  }
  return v;
}

g4_tensor *required_tensor(const gguf *g, const char *t_name) {
  g4_tensor *t = find_tensor(g, t_name);
  if (!t) {
    fprintf(stderr, "Error: missing required tensor: %s\n", t_name);
    exit(1);
  }
  return t;
}
