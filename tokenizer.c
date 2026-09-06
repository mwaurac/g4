#include "tokenizer.h"
#include "gguf.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void bb_init(byte_buf *b) {
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

void bb_reserve(byte_buf *b, size_t extra) {
  if (b->len + extra <= b->capacity)
    return;
  size_t new_cap = b->capacity ? b->capacity * 2 : 64;
  while (new_cap < b->len + extra)
    new_cap *= 2;
  b->data     = (uint8_t *)realloc(b->data, new_cap);
  b->capacity = new_cap;
}

void bb_push(byte_buf *b, const uint8_t *bytes, size_t n) {
  bb_reserve(b, n);
  memcpy(b->data + b->len, bytes, n);
  b->len += n;
}

void bb_push_byte(byte_buf *b, uint8_t byte) {
  bb_push(b, &byte, 1);
}

void bb_free(byte_buf *b) {
  free(b->data);
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

void ib_init(id_buf *b) {
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

void ib_push(id_buf *b, int32_t id) {
  if (b->len == b->capacity) {
    uint64_t new_cap = b->capacity ? b->capacity * 2 : 64;
    b->data          = (int32_t *)realloc(b->data, new_cap * sizeof(int32_t));
    b->capacity      = new_cap;
  }
  b->data[b->len++] = id;
}

void ib_free(id_buf *b) {
  free(b->data);
  b->data     = NULL;
  b->len      = 0;
  b->capacity = 0;
}

static uint64_t next_pow2(uint64_t n) {
  uint64_t p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

void table_init(str_i32_table *t, uint64_t expected_size) {
  t->capacity = next_pow2(expected_size * 2 + 16);
  t->len      = 0;
  t->entries  = xcalloc((size_t)t->capacity, sizeof(str_i32_entry));
}

void table_free(str_i32_table *t) {
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
void table_put(str_i32_table *t, gguf_str key, int32_t value) {
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

bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
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

void tokenizer_free(g4_tokenizer *tok) {
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

g4_tokenizer *load_tokenizer(gguf *g) {
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
  tok->audio_placeholder_id         = table_lookup(tok, "<audio|>");

  return tok;
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
      rl = len - i; // truncated sequence: take what's left
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

id_buf g4_encode(const g4_tokenizer *tok, const char *text, bool add_bos) {
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

static int hexval(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  return -1;
}

static int parse_byte_fallback(const char *ptr, uint64_t len) {
  if (len != 6 || ptr[0] != '<' || ptr[1] != '0' || ptr[2] != 'x' || ptr[5] != '>')
    return -1;
  int hi = hexval(ptr[3]);
  int lo = hexval(ptr[4]);
  if (hi < 0 || lo < 0)
    return -1;
  return (hi << 4) | lo;
}

char *g4_decode(const g4_tokenizer *tok, const int32_t *ids, uint64_t num_ids, bool skip_special) {
  byte_buf raw;
  bb_init(&raw);

  for (uint64_t i = 0; i < num_ids; i++) {
    int32_t id = ids[i];
    if (id < 0 || (uint32_t)id >= tok->vocab_size)
      continue;
    if (skip_special &&
        (id == (int32_t)tok->bos_token_id || id == (int32_t)tok->eos_token_id || id == (int32_t)tok->unk_token_id))
      continue;

    gguf_str piece    = tok->tokens[id];
    int      byte_val = parse_byte_fallback(piece.ptr, piece.len);
    if (byte_val >= 0) {
      bb_push_byte(&raw, (uint8_t)byte_val);
    } else {
      bb_push(&raw, (const uint8_t *)piece.ptr, (size_t)piece.len);
    }
  }

  byte_buf out;
  bb_init(&out);
  // Replace every "▁" marker with a space
  uint64_t i = 0;
  while (i < raw.len) {
    if (i + 3 <= raw.len && memcmp(raw.data + i, SPACE_MARKER_BYTES, 3) == 0) {
      bb_push_byte(&out, ' ');
      i += 3;
    } else {
      bb_push_byte(&out, raw.data[i]);
      i += 1;
    }
  }
  bb_free(&raw);

  // Undo the dummy prefix by dropping space at beginning
  uint64_t start  = (out.len > 0 && out.data[0] == ' ') ? 1 : 0;
  char    *result = (char *)xmalloc((size_t)(out.len - start) + 1);
  memcpy(result, out.data + start, (size_t)(out.len - start));
  result[out.len - start] = '\0';
  bb_free(&out);

  return result;
}

void print_ids(const g4_tokenizer *tok, const id_buf *ids) {
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
