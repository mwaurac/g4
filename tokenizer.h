#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* A dynamic byte buffer */
typedef struct {
  uint8_t *data;
  uint64_t len;
  uint64_t capacity;
} byte_buf;

typedef struct {
  int32_t *data;
  uint64_t len;
  uint64_t capacity;
} id_buf;

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

void bb_init(byte_buf *b);
void bb_reserve(byte_buf *b, size_t extra);
void bb_push(byte_buf *b, const uint8_t *bytes, size_t n);
void bb_push_byte(byte_buf *b, uint8_t byte);
void bb_free(byte_buf *b);

void ib_init(id_buf *b);
void ib_push(id_buf *b, int32_t id);
void ib_free(id_buf *b);

void table_init(str_i32_table *t, uint64_t expected_size);
void table_free(str_i32_table *t);
void table_put(str_i32_table *t, gguf_str key, int32_t value);
bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value);

void tokenizer_free(g4_tokenizer *tok);
g4_tokenizer *load_tokenizer(gguf *g);
id_buf g4_encode(const g4_tokenizer *tok, const char *text, bool add_bos);
char *g4_decode(const g4_tokenizer *tok, const int32_t *ids, uint64_t num_ids, bool skip_special);
void print_ids(const g4_tokenizer *tok, const id_buf *ids);

#endif
