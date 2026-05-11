
/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

static void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

struct ds4_engine {
    ds4_model model;
    ds4_model mtp_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_backend backend;
    int mtp_draft_tokens;
    float mtp_margin;
    bool quality;
    bool metal_ready;
    bool mtp_ready;
};

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
static void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
}

static void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        token_vec       *out) {
    token_vec_push(out, vocab->bos_id);
    if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
    if (system && system[0]) {
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                         token_vec *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    token_vec_push(tokens, e->vocab.bos_id);
}

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        bpe_tokenize_text(vocab, content, tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            bpe_tokenize_text(vocab, "Tool: ", tokens);
        }
        bpe_tokenize_text(vocab, content, tokens);
    }
}

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}

static void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

/* CPU generation entry point.  It runs layer-major prefill once, then decodes
 * one token at a time using the persistent KV cache and scratch arena. */
static int generate_raw_swa_cpu(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    (void)progress;
    (void)progress_ud;
    fprintf(stderr, "ds4: using CPU generation with layer-major prefill\n");

    ds4_kv_cache cache;
    kv_cache_init(&cache, (uint32_t)ctx_size, 0);
    ds4_cpu_decode_scratch decode_scratch;
    cpu_decode_scratch_init(&decode_scratch, (uint32_t)ctx_size);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    int pos = prompt->len;
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const double t_prefill0 = now_sec();

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        free(logits);
        cpu_decode_scratch_free(&decode_scratch);
        kv_cache_free(&cache);
        return 1;
    }

    prefill_layer_major_cpu(logits, model, weights, &cache, prompt);

    const double t_prefill1 = now_sec();
    fprintf(stderr, "ds4: prefill %d/%d done\n", prompt->len, prompt->len);
    const char *dump_prefill_logits = getenv("DS4_CPU_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            cpu_decode_scratch_free(&decode_scratch);
            kv_cache_free(&cache);
            return 1;
        }
        fprintf(stderr, "ds4: wrote CPU prefill logits to %s\n", dump_prefill_logits);
    }

    int n_generated = 0;
    int n_decode_eval = 0;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    ds4_alloc_guard_begin("CPU token generation");
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, &cache, token, (uint32_t)pos,
                                                 &decode_scratch);
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    ds4_alloc_guard_end();
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    free(logits);
    cpu_decode_scratch_free(&decode_scratch);
    kv_cache_free(&cache);
    return 0;
}

#ifndef DS4_NO_METAL
/* Metal generation entry point.  The model runs as one local whole-graph
 * pipeline: chunked/layer-major prefill followed by graph decode steps. */
static int generate_metal_graph_raw_swa(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using Metal graph generation with layer-major graph prefill\n");

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }

    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(prompt->len);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);
    if (prefill_cap < (uint32_t)prompt->len) {
        fprintf(stderr,
                "ds4: using chunked Metal prefill (%u-token chunks for %d prompt tokens)\n",
                prefill_cap,
                prompt->len);
    }
    ds4_metal_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate Metal graph runtime\n");
        return 1;
    }
    g.quality = quality;
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_metal_print_memory_report("after graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;

    const double t_prefill0 = now_sec();
    if (prefill_cap < (uint32_t)prompt->len) {
        ok = metal_graph_prefill_chunked(&g, model, weights, prompt, prompt->len, logits, false, progress, progress_ud);
    } else {
        ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, prompt->len, logits, true);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_metal_print_memory_report("after prefill");

    if (!ok) {
        free(logits);
        metal_graph_free(&g);
        return 1;
    }
    const char *dump_prefill_logits = getenv("DS4_METAL_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            metal_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "ds4: wrote Metal prefill logits to %s\n", dump_prefill_logits);
    }

    int pos = prompt->len;
    int n_generated = 0;
    int n_decode_eval = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = metal_graph_eval_token_raw_swa(&g,
                                            model,
                                            weights,
                                            (uint32_t)token,
                                            (uint32_t)pos,
                                            logits);
        if (!ok) break;
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: metal decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_metal_print_memory_report("before graph free");
    free(logits);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}
#endif

#ifdef DS4_NO_METAL
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    (void)backend;
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    m.raw_cap = ds4_default_raw_cap(ctx);
    m.raw_bytes = (uint64_t)DS4_N_LAYER *
                  m.raw_cap *
                  DS4_N_HEAD_DIM *
                  sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t comp_cap = ctx / ratio + 2u;
        if (ratio == 4) m.comp_cap = comp_cap;
        m.compressed_bytes += (uint64_t)comp_cap *
                              DS4_N_HEAD_DIM *
                              sizeof(float);
        if (ratio == 4) {
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_INDEXER_HEAD_DIM *
                                  sizeof(float);
        }
    }
    if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
    m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(bool));
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}
#endif

/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

const char *ds4_backend_name(ds4_backend backend) {
    return backend == DS4_BACKEND_METAL ? "metal" : "cpu";
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}

struct ds4_session {
    ds4_engine *engine;
#ifndef DS4_NO_METAL
    ds4_metal_graph graph;
#endif
    token_vec checkpoint;
    float *logits;
    float *mtp_logits;
    int mtp_draft_token;
    uint64_t mtp_probe_total;
    uint64_t mtp_probe_hit;
    ds4_session_progress_fn progress;
    void *progress_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool mtp_draft_valid;
};

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

#ifndef DS4_NO_METAL
/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_metal_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_metal_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Metal tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_metal_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_metal_tensor_bytes(tensor) ||
        bytes > ds4_metal_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_metal_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read Metal session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_metal_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_metal_tensor_bytes(tensor) ||
        bytes > ds4_metal_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_metal_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}
#endif

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    const ds4_tensor *gate = e->weights.layer[0].ffn_gate_exps;
    if (!gate) return 0;
    return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
}

bool ds4_engine_has_mtp(ds4_engine *e) {
    return e && e->mtp_ready;
}

int ds4_engine_mtp_draft_tokens(ds4_engine *e) {
    return e && e->mtp_ready ? e->mtp_draft_tokens : 0;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}

#ifndef DS4_NO_METAL
typedef struct {
    uint32_t n_comp[DS4_N_LAYER];
    uint32_t n_index_comp[DS4_N_LAYER];
    uint32_t mtp_n_raw;
} ds4_spec_frontier;

static void spec_frontier_free(ds4_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}

static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s) {
    memset(f, 0, sizeof(*f));
    ds4_metal_graph *g = &s->graph;
    f->mtp_n_raw = g->mtp_n_raw;

    bool ok = ds4_metal_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        f->n_comp[il] = g->layer_n_comp[il];
        f->n_index_comp[il] = g->layer_n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_metal_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_metal_tensor_copy(g->spec_attn_state_kv[il], 0,
                                   g->layer_attn_state_kv[il], 0, ab) != 0 &&
             ds4_metal_tensor_copy(g->spec_attn_state_score[il], 0,
                                   g->layer_attn_state_score[il], 0, ab) != 0;
        if (ratio == 4) {
            const uint64_t ib = ds4_metal_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ok &&
                 ds4_metal_tensor_copy(g->spec_index_state_kv[il], 0,
                                       g->layer_index_state_kv[il], 0, ib) != 0 &&
                 ds4_metal_tensor_copy(g->spec_index_state_score[il], 0,
                                       g->layer_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_metal_end_commands() != 0;
    else (void)ds4_metal_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}

static bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s) {
    ds4_metal_graph *g = &s->graph;
    bool ok = ds4_metal_begin_commands() != 0;
    g->mtp_n_raw = f->mtp_n_raw;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = f->n_comp[il];
        g->layer_n_index_comp[il] = f->n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_metal_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_metal_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_attn_state_kv[il], 0, ab) != 0 &&
             ds4_metal_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            const uint64_t ib = ds4_metal_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_metal_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_index_state_kv[il], 0, ib) != 0 &&
                 ds4_metal_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_metal_end_commands() != 0;
    else (void)ds4_metal_synchronize();
    return ok;
}

/* Commit the prefix-1 state captured by the N=2 speculative verifier.
 *
 * The verifier has already advanced every layer through both draft tokens.  On
 * a one-token accept the append-only compressed caches can keep the second
 * speculative row as invisible garbage, but the compressor frontiers and row
 * counters must be rewound to the exact state after draft[0].  This is the
 * cheap partial-accept path: copy a few small per-layer frontiers instead of
 * restoring the whole prefix and replaying a one-token target decode. */
static bool spec_frontier_commit_prefix1(ds4_session *s) {
    ds4_metal_graph *g = &s->graph;
    bool ok = ds4_metal_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;

        g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];
        const uint64_t ab = ds4_metal_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_metal_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_prefix1_attn_state_kv[il], 0, ab) != 0 &&
             ds4_metal_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_prefix1_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            g->layer_n_index_comp[il] = g->spec_prefix1_n_index_comp[il];
            const uint64_t ib = ds4_metal_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_metal_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_prefix1_index_state_kv[il], 0, ib) != 0 &&
                 ds4_metal_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_prefix1_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_metal_end_commands() != 0;
    else (void)ds4_metal_synchronize();
    return ok;
}
#endif

uint64_t ds4_session_payload_bytes(ds4_session *s) {
#ifdef DS4_NO_METAL
    (void)s;
    return 0;
#else
    if (!s || !s->checkpoint_valid) return 0;
    const ds4_metal_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
#endif
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s; (void)fp;
    payload_set_err(err, errlen, "Metal support is not compiled in");
    return 1;
#else
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (ds4_metal_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize Metal before snapshot");
        return 1;
    }

    ds4_metal_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        rc = payload_write_tensor_span(fp,
                                       g->layer_attn_comp_cache[il],
                                       0,
                                       (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                       buf,
                                       DS4_SESSION_IO_CHUNK,
                                       err,
                                       errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s; (void)fp; (void)payload_bytes;
    payload_set_err(err, errlen, "Metal support is not compiled in");
    return 1;
#else
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    ds4_metal_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    if (saved_prefill_cap != s->prefill_cap || saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_N_LAYER];
    uint32_t n_index_comp[DS4_N_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->comp_cap) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->comp_cap) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_read_tensor_span(fp,
                                      g->layer_attn_comp_cache[il],
                                      0,
                                      (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                      buf,
                                      DS4_SESSION_IO_CHUNK,
                                      &remaining,
                                      err,
                                      errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;
    return 0;
#endif
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (e->backend == DS4_BACKEND_METAL) {
#ifndef DS4_NO_METAL
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: Metal generation requested but Metal is unavailable\n");
            return 1;
        }
        return generate_metal_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality, emit, done, emit_ud,
                                            progress, progress_ud);
#else
        fprintf(stderr, "ds4: Metal generation requested but this build has no Metal support\n");
        return 1;
#endif
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size, emit, done, emit_ud, progress, progress_ud);
}

int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_METAL
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_decode_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_METAL
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal full graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_first_token_full_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal full graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
#ifndef DS4_NO_METAL
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal prompt graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
#else
    (void)e;
    (void)prompt;
    (void)ctx_size;
    fprintf(stderr, "ds4: Metal prompt graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1], true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > 16) e->mtp_draft_tokens = 16;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 3.0f;
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    model_open(&e->model, opt->model_path,
               opt->backend == DS4_BACKEND_METAL, true);
    if (opt->warm_weights) model_warm_weights(&e->model);
    vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model);
    if (opt->mtp_path && opt->mtp_path[0]) {
        model_open(&e->mtp_model, opt->mtp_path,
                   opt->backend == DS4_BACKEND_METAL, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
        e->mtp_ready = true;
        fprintf(stderr, "ds4: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
    }

#ifndef DS4_NO_METAL
    if (e->backend == DS4_BACKEND_METAL) {
        e->metal_ready = ds4_metal_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: Metal backend unavailable; aborting startup\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_metal_set_quality(e->quality);
        if (!ds4_metal_set_model_map_range(e->model.map,
                                           e->model.size,
                                           e->model.tensor_data_pos,
                                           e->model.size - e->model.tensor_data_pos))
        {
            fprintf(stderr,
                    "ds4: Metal failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or Metal VM budget.\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready &&
            !ds4_metal_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos))
        {
            fprintf(stderr,
                    "ds4: Metal failed to map MTP model views; aborting startup. "
                    "This is commonly caused by insufficient memory or Metal VM budget.\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "ds4: Metal backend initialized for graph diagnostics\n");
    }
#else
    if (e->backend == DS4_BACKEND_METAL) {
        fprintf(stderr, "ds4: Metal backend requested but this build has no Metal support; aborting startup\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->mtp_ready) model_close(&e->mtp_model);
    model_close(&e->model);
#ifndef DS4_NO_METAL
    ds4_metal_cleanup();
#endif
    ds4_release_instance_lock();
    free(e);
}

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
#ifdef DS4_NO_METAL
    (void)out;
    (void)e;
    (void)ctx_size;
    return 1;
#else
    if (e->backend != DS4_BACKEND_METAL || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, &e->weights.layer[0],
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap, e->mtp_ready))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
    *out = s;
    return 0;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
#ifndef DS4_NO_METAL
    metal_graph_free(&s->graph);
#endif
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}

/* Bring the Metal graph to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "Metal support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    if (prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = metal_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            ds4_session_progress_fn progress_fn =
                s->progress ? ds4_session_note_prefill_progress : NULL;
            bool ok = metal_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        progress_fn,
                                                        progress_fn ? &progress : NULL);
            if (!ok) {
                snprintf(err, errlen, "Metal resumed prefill failed while extending checkpoint");
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "Metal decode failed while extending checkpoint");
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ds4_session_progress_fn progress_fn =
            s->progress ? ds4_session_note_prefill_progress : NULL;
        ok = metal_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         progress_fn, progress_fn ? &progress : NULL);
    } else {
        ok = metal_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false);
    }
    if (!ok) {
        snprintf(err, errlen, "Metal prefill failed");
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the Metal graph also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * graph frontier first.  Extending exactly at the live end is safe; rewriting
 * behind it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the Metal graph still contains raw and compressed
 * rows for the old suffix.  Until we have a real graph frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the graph.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s;
    (void)prompt;
    (void)common;
    snprintf(err, errlen, "Metal support is not compiled in");
    return DS4_SESSION_REWRITE_ERROR;
#else
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
#endif
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int ds4_session_argmax(ds4_session *s) {
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

static int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s;
    (void)token;
    (void)probe_mtp;
    snprintf(err, errlen, "Metal support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const bool mtp_probe_log = getenv("DS4_MTP_PROBE") != NULL;
    const bool mtp_should_draft =
        probe_mtp && e->mtp_ready && s->mtp_logits &&
        (e->mtp_draft_tokens > 1 || mtp_probe_log);
    if (probe_mtp && s->mtp_draft_valid) {
        if (mtp_probe_log) {
            s->mtp_probe_total++;
            if (s->mtp_draft_token == token) s->mtp_probe_hit++;
            fprintf(stderr,
                    "ds4: mtp probe token=%d draft=%d hit=%llu/%llu\n",
                    token,
                    s->mtp_draft_token,
                    (unsigned long long)s->mtp_probe_hit,
                    (unsigned long long)s->mtp_probe_total);
        }
        s->mtp_draft_valid = false;
    }
    if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "Metal decode failed");
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    if (mtp_should_draft) {
        int mtp_top = -1;
        if (metal_graph_eval_mtp_draft(&s->graph,
                                       &e->model,
                                       &e->weights,
                                       &e->mtp_model,
                                       &e->mtp_weights,
                                       token,
                                       (uint32_t)(s->checkpoint.len - 1),
                                       getenv("DS4_MTP_FULL_LOGITS") ? s->mtp_logits : NULL,
                                       &mtp_top)) {
            s->mtp_draft_token = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
            s->mtp_draft_valid = true;
        } else if (getenv("DS4_MTP_PROBE")) {
            fprintf(stderr, "ds4: mtp probe draft failed\n");
        }
    }
    return 0;
#endif
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    return ds4_session_eval_internal(s, token, true, err, errlen);
}

/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
#ifdef DS4_NO_METAL
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "Metal support is not compiled in");
    return -1;
#else
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    ds4_engine *e = s->engine;

    /*
     * MTP in DeepSeek V4 is a speculative drafter, not a replacement sampler.
     * The target model still defines the exact output stream.  A cycle starts
     * by accepting one normal target token, then asks the MTP block to propose
     * a short suffix.  The suffix is useful only if the target model can verify
     * several proposed positions together; running ordinary decode once per
     * draft token is correctness-safe but cannot be faster than baseline.
     */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap) return n_accept;

    if (!e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1) return n_accept;

    int draft_cap = e->mtp_draft_tokens;
    if (draft_cap > max_tokens - n_accept) draft_cap = max_tokens - n_accept;
    if (draft_cap > accepted_cap - n_accept) draft_cap = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (draft_cap > room - 1) draft_cap = room - 1;
    if (draft_cap <= 0) return n_accept;

    int drafts[16];
    int draft_n = 1;
    drafts[0] = s->mtp_draft_token;
    s->mtp_draft_valid = false;
    const bool strict_mtp = e->quality || getenv("DS4_MTP_STRICT") != NULL;
    float mtp_margin_threshold = e->mtp_margin;
    const char *mtp_margin_env = getenv("DS4_MTP_MIN_MARGIN");
    if (mtp_margin_env && mtp_margin_env[0]) {
        char *end = NULL;
        float v = strtof(mtp_margin_env, &end);
        if (end != mtp_margin_env && v >= 0.0f) mtp_margin_threshold = v;
    }
    const bool mtp_timing = getenv("DS4_MTP_TIMING") != NULL;
    const bool mtp_conf_log = getenv("DS4_MTP_CONF_LOG") != NULL;
    const bool mtp_need_logits = mtp_conf_log ||
        getenv("DS4_MTP_FULL_LOGITS") != NULL ||
        (!strict_mtp && mtp_margin_threshold > 0.0f);
    const double mtp_t0 = mtp_timing ? now_sec() : 0.0;
    double mtp_t_after_draft = mtp_t0;
    float mtp_last_margin = 0.0f;
    int mtp_last_top0 = -1, mtp_last_top1 = -1;

    /*
     * The first proposed token is verified for free: ds4_session_eval() just
     * produced the base logits for the committed prefix.  If MTP disagrees at
     * this point there is no suffix to verify, so the exact behavior is to emit
     * only first_token and skip all speculative work.
     */
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec miss first draft=%d\n", drafts[0]);
        }
        return n_accept;
    }
    if (drafts[0] == eos_token) draft_cap = 1;
    const uint32_t mtp_base_raw = s->graph.mtp_n_raw;
    /*
     * MTP has its own raw SWA cache. Recursive drafting writes speculative
     * future rows into it; after verification, rows beyond the accepted prefix
     * must become invisible.  We do not copy/rollback the cache body because the
     * next draft attempt will overwrite future slots.  A counter is enough.
     */
#define DS4_MTP_KEEP_ACCEPTED(n_) do { \
        uint32_t keep_ = mtp_base_raw + (uint32_t)(n_); \
        if (keep_ > s->graph.raw_window) keep_ = s->graph.raw_window; \
        s->graph.mtp_n_raw = keep_; \
    } while (0)

    for (; draft_n < draft_cap; draft_n++) {
        ds4_metal_tensor *prev_hc = (draft_n & 1) ? s->graph.mtp_state_hc : s->graph.mtp_next_hc;
        ds4_metal_tensor *out_hc = (draft_n & 1) ? s->graph.mtp_next_hc : s->graph.mtp_state_hc;
        int mtp_top = -1;
        if (!metal_graph_eval_mtp_draft_from_hc(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &e->mtp_model,
                                                &e->mtp_weights,
                                                prev_hc,
                                                out_hc,
                                                drafts[draft_n - 1],
                                                (uint32_t)(s->checkpoint.len + draft_n - 1),
                                                mtp_need_logits ? s->mtp_logits : NULL,
                                                &mtp_top))
        {
            return n_accept;
        }
        drafts[draft_n] = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
        if (drafts[draft_n] == eos_token) {
            draft_n++;
            break;
        }
    }
    if (mtp_conf_log && draft_n > 1) {
        float v0 = 0.0f, v1 = 0.0f;
        logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
        mtp_last_margin = v0 - v1;
    }
    if (mtp_timing) mtp_t_after_draft = now_sec();

    if (!strict_mtp && draft_n == 2 && mtp_margin_threshold > 0.0f) {
        if (!mtp_conf_log) {
            float v0 = 0.0f, v1 = 0.0f;
            logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
            mtp_last_margin = v0 - v1;
        }
        if (mtp_last_margin < mtp_margin_threshold) {
            float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
            const int start = s->checkpoint.len;
            const double verify_t0 = mtp_timing ? now_sec() : 0.0;
            bool ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                     &e->model,
                                                     &e->weights,
                                                     drafts[0],
                                                     (uint32_t)start,
                                                     row_logits);
            if (!ok) {
                free(row_logits);
                snprintf(err, errlen, "Metal decode failed");
                s->checkpoint_valid = false;
                return -1;
            }
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            free(row_logits);
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing margin-skip drafted=2 committed=1 margin=%.3f threshold=%.3f draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        mtp_last_margin,
                        mtp_margin_threshold,
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (done - verify_t0) * 1000.0,
                        (done - mtp_t0) * 1000.0);
            }
            return n_accept;
        }
    }

    /*
     * The useful N=2 verifier is the tiny batch path: it verifies two target
     * positions in one layer-major pass and commits prefix-1 directly on a
     * partial accept.  Like the rest of the non-quality Metal path, it may pick
     * a different greedy token when batched reductions perturb nearly-tied
     * logits.  --quality / DS4_MTP_STRICT selects the exact decode verifier,
     * which preserves the one-token target stream but is not a speed win.
     */
    const bool use_decode2_exact =
        draft_n == 2 && strict_mtp && getenv("DS4_MTP_BATCH_VERIFY") == NULL;
    if (use_decode2_exact) {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        float *row0_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row0_logits[0]));
        const int start = s->checkpoint.len;
        int row0_top = -1;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        bool have_frontier = spec_frontier_snapshot(&frontier, s);
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        bool ok = have_frontier;
        if (ok) {
            ok = metal_graph_verify_decode2_exact(&s->graph,
                                                  &e->model,
                                                  &e->weights,
                                                  drafts[0],
                                                  drafts[1],
                                                  (uint32_t)start,
                                                  &row0_top,
                                                  row0_logits,
                                                  row_logits);
        }
        const double verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok && row0_top == drafts[1]) {
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            token_vec_push(&s->checkpoint, drafts[0]);
            token_vec_push(&s->checkpoint, drafts[1]);
            accepted[n_accept++] = drafts[0];
            if (n_accept < accepted_cap) accepted[n_accept++] = drafts[1];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(2);
            if (mtp_timing) {
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=2 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (now_sec() - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }

        if (ok) {
            s->checkpoint.len = start;
            ok = spec_frontier_commit_prefix1(s);
        }
        if (ok) memcpy(s->logits, row0_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        if (ok) {
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double replay_done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=1 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (replay_done - verify_done) * 1000.0,
                        (replay_done - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }
        if (have_frontier) {
            s->checkpoint.len = start;
            (void)spec_frontier_restore(&frontier, s);
        }
        spec_frontier_free(&frontier);
        free(row0_logits);
        free(row_logits);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp decode2 verifier failed, falling back to sequential\n");
        }
    }

    if (!use_decode2_exact)
    {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        int *row_tops = xmalloc((size_t)draft_n * sizeof(row_tops[0]));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        const int start = s->checkpoint.len;
        /*
         * The production MTP depth is two.  Prefix-1 capture makes partial
         * accepts cheap, but it copies per-layer compressor frontiers even when
         * both draft tokens are accepted.  Full accepts are the path that makes
         * MTP worthwhile, so by default we snapshot before the verifier and
         * replay one token on partial accept.  DS4_MTP_CAPTURE_PREFIX1 restores
         * the older no-replay partial path for measurement.
         */
        const bool capture_prefix1 =
            draft_n == 2 && (!strict_mtp || getenv("DS4_MTP_CAPTURE_PREFIX1") != NULL);
        const bool exact_replay_debug = getenv("DS4_MTP_EXACT_REPLAY") != NULL;
        const bool snapshot_required =
            draft_n > 2 ||
            (draft_n == 2 && (!capture_prefix1 || exact_replay_debug)) ||
            getenv("DS4_MTP_FORCE_SNAPSHOT") != NULL;
        bool have_frontier = false;
        bool ok = true;
        bool verifier_may_have_mutated = false;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        if (snapshot_required) {
            have_frontier = spec_frontier_snapshot(&frontier, s);
            ok = have_frontier;
        }
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
            verifier_may_have_mutated = true;
            ok = metal_graph_verify_suffix_tops(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &s->checkpoint,
                                                (uint32_t)start,
                                                (uint32_t)draft_n,
                                                capture_prefix1,
                                                row_tops,
                                                NULL);
        }
        const double micro_verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            int commit_drafts = 1;
            for (int i = 1; i < draft_n; i++) {
                if (row_tops[i - 1] != drafts[i]) break;
                commit_drafts++;
            }
            if (mtp_conf_log) {
                fprintf(stderr,
                        "ds4: mtp conf drafted=%d committed=%d mtp_top=%d runner=%d margin=%.6f target_next=%d draft_next=%d\n",
                        draft_n,
                        commit_drafts,
                        mtp_last_top0,
                        mtp_last_top1,
                        mtp_last_margin,
                        draft_n > 1 ? row_tops[0] : -1,
                        draft_n > 1 ? drafts[1] : -1);
            }
            if (exact_replay_debug && have_frontier) {
                s->checkpoint.len = start;
                ok = spec_frontier_restore(&frontier, s);
                if (ok) {
                    int replayed = 0;
                    for (; replayed < commit_drafts && ok; replayed++) {
                        ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                            &e->model,
                                                            &e->weights,
                                                            drafts[replayed],
                                                            (uint32_t)(start + replayed),
                                                            row_logits);
                        if (ok) token_vec_push(&s->checkpoint, drafts[replayed]);
                    }
                    if (ok) {
                        memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                        for (int i = 0; i < replayed && n_accept < accepted_cap; i++) {
                            accepted[n_accept++] = drafts[i];
                            if (drafts[i] == eos_token) break;
                        }
                        s->checkpoint_valid = true;
                        s->mtp_draft_valid = false;
                        DS4_MTP_KEEP_ACCEPTED(replayed);
                        spec_frontier_free(&frontier);
                        free(row_logits);
                        free(row_tops);
                        return n_accept;
                    }
                }
            }

            if (commit_drafts == draft_n) {
                ok = metal_graph_read_spec_logits_row(&s->graph,
                                                      (uint32_t)(draft_n - 1),
                                                      row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(draft_n);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                                draft_n,
                                draft_n,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }

            if (draft_n == 2 && commit_drafts == 1 && capture_prefix1) {
                s->checkpoint.len = start;
                const double prefix_t0 = mtp_timing ? now_sec() : 0.0;
                ok = spec_frontier_commit_prefix1(s);
                const double prefix_done = mtp_timing ? now_sec() : 0.0;
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph, 0, row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms noreplay=1\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (prefix_done - prefix_t0) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            } else {
                s->checkpoint.len = start;
                ok = have_frontier && spec_frontier_restore(&frontier, s);
            }
            if (ok && draft_n == 2 && commit_drafts == 1) {
                ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    drafts[0],
                                                    (uint32_t)start,
                                                    row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms exact_replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
            if (ok) {
                for (int i = 0; i < commit_drafts; i++) token_vec_push(&s->checkpoint, drafts[i]);
                ok = metal_graph_verify_suffix_tops(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)start,
                                                    (uint32_t)commit_drafts,
                                                    false,
                                                    row_tops,
                                                    NULL);
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph,
                                                              (uint32_t)(commit_drafts - 1),
                                                              row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(commit_drafts);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
        }
        s->checkpoint.len = start;
        if (have_frontier) {
            (void)spec_frontier_restore(&frontier, s);
        } else if (!verifier_may_have_mutated) {
            /* Snapshot setup failed before the verifier touched Metal state.
             * Fall through to the exact sequential verifier below. */
        } else {
            snprintf(err, errlen, "MTP verifier failed");
            s->checkpoint_valid = false;
            DS4_MTP_KEEP_ACCEPTED(0);
            spec_frontier_free(&frontier);
            free(row_logits);
            free(row_tops);
            return -1;
        }
        spec_frontier_free(&frontier);
        free(row_logits);
        free(row_tops);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec micro verifier failed, falling back to sequential\n");
        }
    }

    /*
     * Safety fallback: if the production microbatch verifier fails, verify
     * drafts with the exact normal one-token decode path instead of returning
     * wrong state.  This path is deliberately slow and should not be selected
     * during normal --mtp operation.
     */
    int verified = 0;
    int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
    bool logits_on_host = true;
    const double seq_t0 = mtp_timing ? now_sec() : 0.0;
    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
        if (target_top != drafts[i]) {
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: mtp spec seq miss at=%d draft=%d base=%d drafted=%d accepted=%d\n",
                        i,
                        drafts[i],
                        target_top,
                        draft_n,
                        n_accept);
            }
            break;
        }
        if (!metal_graph_eval_token_raw_swa_top(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                drafts[i],
                                                (uint32_t)s->checkpoint.len,
                                                &target_top,
                                                NULL))
        {
            snprintf(err, errlen, "Metal decode failed");
            s->checkpoint_valid = false;
            return -1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
        logits_on_host = false;
        accepted[n_accept++] = drafts[i];
        verified++;
        if (drafts[i] == eos_token) break;
    }
    if (verified > 0 && !logits_on_host) {
        if (ds4_metal_tensor_read(s->graph.logits,
                                  0,
                                  s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0)
        {
            snprintf(err, errlen, "Metal logits readback failed");
            s->checkpoint_valid = false;
            return -1;
        }
        logits_on_host = true;
    }
    (void)logits_on_host;
    DS4_MTP_KEEP_ACCEPTED(verified);
#undef DS4_MTP_KEEP_ACCEPTED
    if (mtp_timing) {
        fprintf(stderr,
                "ds4: mtp timing seq drafted=%d verified=%d draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                draft_n,
                verified,
                (mtp_t_after_draft - mtp_t0) * 1000.0,
                (now_sec() - seq_t0) * 1000.0,
                (now_sec() - mtp_t0) * 1000.0);
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        if (verified == draft_n) {
            fprintf(stderr,
                    "ds4: mtp spec seq accept drafted=%d accepted=%d\n",
                    draft_n,
                    n_accept);
        } else {
            fprintf(stderr,
                    "ds4: mtp spec seq partial drafted=%d verified=%d accepted=%d\n",
                    draft_n,
                    verified,
                    n_accept);
        }
    }
    return n_accept;
#endif
}

void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}
