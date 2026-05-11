
/* Scratch-backed indexer selection for decode. */
static bool *indexer_allowed_decode_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos,
        ds4_cpu_decode_scratch  * scratch) {
    if (n_comp == 0) return NULL;
    if (n_comp > scratch->comp_cap) ds4_die("CPU decode indexer scratch buffer is too small");

    bool *allowed = scratch->index_allowed;
    memset(allowed, 0, (size_t)n_comp * sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = scratch->index_q;
    float *weights = scratch->index_weights;
    float *scores = scratch->index_scores;

    matvec_any_decode_scratch(q, model, layer->indexer_attn_q_b, qr_norm, scratch);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);

    matvec_any_decode_scratch(weights, model, layer->indexer_proj, cur, scratch);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    return allowed;
}

/* Single-token attention sublayer with raw SWA cache and DS4 compression. */
static void layer_attention_raw_swa_one(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *qr_norm = xmalloc(1024 * sizeof(qr_norm[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_with_lora_one(model, layer, attn_norm, q, qr_norm);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);

    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, kv);

    const uint32_t ratio = cache->compress_ratio;
    if (ratio != 0) {
        float *comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp[0]));
        if (compressor_decode_one(comp, model,
                                  layer->attn_compressor_kv,
                                  layer->attn_compressor_gate,
                                  layer->attn_compressor_ape,
                                  layer->attn_compressor_norm,
                                  attn_norm,
                                  cache->attn_state_kv,
                                  cache->attn_state_score,
                                  DS4_N_HEAD_DIM,
                                  ratio,
                                  il,
                                  pos)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
        }
        free(comp);

        if (ratio == 4) {
            float *index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp[0]));
            if (compressor_decode_one(index_comp, model,
                                      layer->indexer_compressor_kv,
                                      layer->indexer_compressor_gate,
                                      layer->indexer_compressor_ape,
                                      layer->indexer_compressor_norm,
                                      attn_norm,
                                      cache->index_state_kv,
                                      cache->index_state_score,
                                      DS4_N_INDEXER_HEAD_DIM,
                                      ratio,
                                      il,
                                      pos)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
            }
            free(index_comp);

            comp_allowed = indexer_allowed_decode_one(model, layer,
                                                      attn_norm, qr_norm,
                                                      cache->index_comp_kv,
                                                      cache->n_index_comp,
                                                      il, pos);
        }

        layer_attention_mixed_one(heads, model, layer, q,
                                  cache->raw_kv, cache->n_raw,
                                  cache->attn_comp_kv, cache->n_comp,
                                  comp_allowed);
    } else {
        layer_attention_rows_one(heads, model, layer, q, cache->raw_kv, cache->n_raw);
    }

    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    free(comp_allowed);
    free(attn_out);
    free(heads);
    free(kv);
    free(qr_norm);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Batched prefill attention.  It projects Q/KV for all tokens, streams them
 * through the same raw/compressed cache updates, then runs prefix attention. */
static void layer_attention_raw_swa_batch(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  n_tok,
        uint32_t                  il,
        uint32_t                  pos0) {
    const bool profile = getenv("DS4_PREFILL_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_token_loop = 0.0;
    double t_tl_rope_cache = 0.0;
    double t_tl_compress = 0.0;
    double t_tl_indexer = 0.0;
    double t_tl_attn_rows = 0.0;
    double t_tl_inv_rope = 0.0;
    double t_out = 0.0;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_tok * hc_dim * sizeof(attn_residual[0]));
    float *qr = xmalloc((size_t)n_tok * 1024 * sizeof(qr[0]));
    float *qr_norm = xmalloc((size_t)n_tok * 1024 * sizeof(qr_norm[0]));
    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv_raw = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv_raw[0]));
    float *kv = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = NULL;
    float *attn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_out[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_attn_fn,
                      layer->hc_attn_scale,
                      layer->hc_attn_base,
                      layer->attn_norm,
                      inp_hc,
                      attn_residual,
                      attn_cur,
                      attn_norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(qr, model, layer->attn_q_a, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(qr_norm + (uint64_t)t * 1024,
                        qr + (uint64_t)t * 1024,
                        q_a_norm,
                        1024,
                        DS4_RMS_EPS);
    }
    matmul_q8_0_batch(q, model, layer->attn_q_b, qr_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        head_rms_norm_inplace(q + (uint64_t)t * q_dim,
                              DS4_N_HEAD,
                              DS4_N_HEAD_DIM,
                              DS4_RMS_EPS);
    }
    if (profile) t_q = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(kv_raw, model, layer->attn_kv, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(kv + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_raw + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_norm,
                        DS4_N_HEAD_DIM,
                        DS4_RMS_EPS);
    }
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    const uint32_t ratio = cache->compress_ratio;
    const bool prefer_parallel_attn = getenv("DS4_PARALLEL_ATTN_ROWS") != NULL;
    const bool prefix_batch_attn =
        prefer_parallel_attn &&
        getenv("DS4_NO_PARALLEL_ATTN_ROWS") == NULL &&
        cache->n_raw == 0 &&
        pos0 == 0;
    if (!prefix_batch_attn) {
        heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
    }
    uint32_t batch_rope_max = 4096;
    const char *batch_rope_max_env = getenv("DS4_BATCHED_ROPE_MAX");
    if (batch_rope_max_env && batch_rope_max_env[0]) {
        long v = strtol(batch_rope_max_env, NULL, 10);
        if (v >= 0 && v <= 65536) batch_rope_max = (uint32_t)v;
    }
    const bool batch_prefix_rope =
        prefix_batch_attn &&
        getenv("DS4_NO_BATCHED_ROPE") == NULL &&
        n_tok <= batch_rope_max;
    uint32_t *comp_counts = prefix_batch_attn ?
        xcalloc((size_t)n_tok, sizeof(comp_counts[0])) : NULL;
    uint8_t *allowed_mask = prefix_batch_attn && ratio == 4 ?
        xcalloc((size_t)n_tok, sizeof(allowed_mask[0])) : NULL;
    uint8_t *allowed_bits = NULL;
    const uint64_t allowed_stride = ratio == 4 ? ((uint64_t)cache->comp_cap + 7u) / 8u : 0;
    float *comp_scratch = NULL;
    float *index_comp_scratch = NULL;

    if (ratio != 0) {
        comp_scratch = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp_scratch[0]));

        if (ratio == 4) {
            index_comp_scratch = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp_scratch[0]));
        }
    }

    if (batch_prefix_rope) {
        double tx = profile ? now_sec() : 0.0;
        rope_tail_layer_batch_inplace(q,
                                      q_dim,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        rope_tail_layer_batch_inplace(kv,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_HEAD_KV,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        if (profile) t_tl_rope_cache += now_sec() - tx;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        float *q_t = q + (uint64_t)t * q_dim;
        float *kv_t = kv + (uint64_t)t * DS4_N_HEAD_DIM;
        bool *comp_allowed = NULL;

        double tx = profile ? now_sec() : 0.0;
        if (!batch_prefix_rope) {
            rope_tail_layer_inplace(q_t, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
            rope_tail_layer_inplace(kv_t, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
        }
        dsv4_fp8_kv_quantize_row_inplace_cpu(kv_t, DS4_N_HEAD_DIM, DS4_N_ROT);

        kv_cache_push_raw(cache, kv_t);
        if (profile) t_tl_rope_cache += now_sec() - tx;

        if (ratio != 0) {
            tx = profile ? now_sec() : 0.0;
            float *comp = comp_scratch;
            const bool have_comp = compressor_decode_one(comp, model,
                                                         layer->attn_compressor_kv,
                                                         layer->attn_compressor_gate,
                                                         layer->attn_compressor_ape,
                                                         layer->attn_compressor_norm,
                                                         attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                         cache->attn_state_kv,
                                                         cache->attn_state_score,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         il,
                                                         pos);
            if (have_comp) {
                kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
            }

            if (ratio == 4) {
                float *index_comp = index_comp_scratch;
                const bool have_index_comp = compressor_decode_one(index_comp, model,
                                                                   layer->indexer_compressor_kv,
                                                                   layer->indexer_compressor_gate,
                                                                   layer->indexer_compressor_ape,
                                                                   layer->indexer_compressor_norm,
                                                                   attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                                   cache->index_state_kv,
                                                                   cache->index_state_score,
                                                                   DS4_N_INDEXER_HEAD_DIM,
                                                                   ratio,
                                                                   il,
                                                                   pos);
                if (have_index_comp) {
                    kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
                }
                if (profile) t_tl_compress += now_sec() - tx;

                tx = profile ? now_sec() : 0.0;
                comp_allowed = indexer_allowed_decode_one(model, layer,
                                                          attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                          qr_norm + (uint64_t)t * 1024,
                                                          cache->index_comp_kv,
                                                          cache->n_index_comp,
                                                          il, pos);
                if (profile) t_tl_indexer += now_sec() - tx;
            } else {
                if (profile) t_tl_compress += now_sec() - tx;
            }

            if (comp_counts) comp_counts[t] = cache->n_comp;
            if (prefix_batch_attn && comp_allowed) {
                if (!allowed_bits) {
                    allowed_bits = xcalloc((size_t)n_tok * allowed_stride, sizeof(allowed_bits[0]));
                }
                allowed_mask[t] = 1;
                uint8_t *bits = allowed_bits + (uint64_t)t * allowed_stride;
                for (uint32_t c = 0; c < cache->n_comp; c++) {
                    if (comp_allowed[c]) bits[c >> 3] |= (uint8_t)(1u << (c & 7u));
                }
            }

            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_mixed_one(heads + (uint64_t)t * q_dim, model, layer, q_t,
                                          cache->raw_kv, cache->n_raw,
                                          cache->attn_comp_kv, cache->n_comp,
                                          comp_allowed);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        } else {
            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_rows_one(heads + (uint64_t)t * q_dim, model, layer, q_t, cache->raw_kv, cache->n_raw);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        }

        if (!prefix_batch_attn) {
            tx = profile ? now_sec() : 0.0;
            rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                    DS4_N_HEAD,
                                    DS4_N_HEAD_DIM,
                                    DS4_N_ROT,
                                    pos,
                                    il,
                                    true);
            if (profile) t_tl_inv_rope += now_sec() - tx;
        }

        free(comp_allowed);
    }

    if (prefix_batch_attn) {
        double tx = profile ? now_sec() : 0.0;
        const float *comp_kv_for_prefix = cache->attn_comp_kv ? cache->attn_comp_kv : kv;
        if (!heads) {
            heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
        }
        layer_attention_prefix_batch(heads, model, layer,
                                     q,
                                     kv,
                                     comp_kv_for_prefix,
                                     comp_counts,
                                     allowed_mask,
                                     allowed_bits,
                                     allowed_stride,
                                     n_tok,
                                     cache->cap_raw);
        if (profile) t_tl_attn_rows += now_sec() - tx;
        tx = profile ? now_sec() : 0.0;
        if (batch_prefix_rope) {
            rope_tail_layer_batch_inplace(heads,
                                          q_dim,
                                          DS4_N_HEAD,
                                          DS4_N_HEAD_DIM,
                                          DS4_N_ROT,
                                          pos0,
                                          il,
                                          true,
                                          n_tok);
        } else {
            for (uint32_t t = 0; t < n_tok; t++) {
                rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                        DS4_N_HEAD,
                                        DS4_N_HEAD_DIM,
                                        DS4_N_ROT,
                                        pos0 + t,
                                        il,
                                        true);
            }
        }
        if (profile) t_tl_inv_rope += now_sec() - tx;
    }
    if (profile) t_token_loop = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_batch(attn_out, model, layer, heads, n_tok);

    hc_post_batch(after_attn_hc,
                  attn_out,
                  attn_residual,
                  post,
                  comb,
                  n_tok,
                  DS4_N_EMBD,
                  n_hc);
    if (profile) t_out = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: prefill detail layer %u attn hc_norm=%.3f q=%.3f kv=%.3f token_loop=%.3f out=%.3f total=%.3f\n",
                il, t_hc_norm, t_q, t_kv, t_token_loop, t_out, now_sec() - t_start);
        if (getenv("DS4_PREFILL_PROFILE_TOKEN") != NULL) {
            fprintf(stderr,
                    "ds4: prefill token detail layer %u rope_cache=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f\n",
                    il, t_tl_rope_cache, t_tl_compress, t_tl_indexer, t_tl_attn_rows, t_tl_inv_rope);
        }
    }

    free(allowed_bits);
    free(allowed_mask);
    free(comp_counts);
    free(index_comp_scratch);
    free(comp_scratch);
    free(comb);
    free(post);
    free(attn_out);
    free(heads);
    free(kv);
    free(kv_raw);
    free(q);
    free(qr_norm);
    free(qr);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Full transformer layer for one decode token: attention sublayer followed by
 * FFN sublayer, both operating on the HC state. */
static void layer_forward_raw_swa_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token,
        ds4_cpu_decode_scratch  * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_rope_cache = 0.0;
    double t_compress = 0.0;
    double t_indexer = 0.0;
    double t_attn_rows = 0.0;
    double t_inv_rope = 0.0;
    double t_out = 0.0;
    double t_post = 0.0;
    double t_ffn = 0.0;

    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    memcpy(scratch->attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_attn_fn,
                                  layer->hc_attn_scale,
                                  layer->hc_attn_base,
                                  scratch->attn_residual, scratch->attn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    if (profile) t_hc = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_attn_norm_one(scratch->attn_norm, model, layer, scratch->attn_cur);
    const uint32_t ratio = cache->compress_ratio;
    layer_q_projection_with_lora_one_decode_scratch(model, layer,
                                                    scratch->attn_norm,
                                                    scratch->q,
                                                    scratch->qr_norm,
                                                    scratch);
    if (profile) t_q = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_kv_projection_normed_one_decode_scratch(model, layer,
                                                  scratch->attn_norm,
                                                  scratch->kv,
                                                  scratch);
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(scratch->kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, scratch->kv);
    if (profile) t_rope_cache = now_sec() - t0;

    if (ratio != 0) {
        t0 = profile ? now_sec() : 0.0;
        if (compressor_decode_one_decode_scratch(scratch->comp, model,
                                                 layer->attn_compressor_kv,
                                                 layer->attn_compressor_gate,
                                                 layer->attn_compressor_ape,
                                                 layer->attn_compressor_norm,
                                                 scratch->attn_norm,
                                                 cache->attn_state_kv,
                                                 cache->attn_state_score,
                                                 DS4_N_HEAD_DIM,
                                                 ratio,
                                                 il,
                                                 pos,
                                                 scratch)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, scratch->comp);
        }

        if (ratio == 4) {
            if (compressor_decode_one_decode_scratch(scratch->index_comp, model,
                                                     layer->indexer_compressor_kv,
                                                     layer->indexer_compressor_gate,
                                                     layer->indexer_compressor_ape,
                                                     layer->indexer_compressor_norm,
                                                     scratch->attn_norm,
                                                     cache->index_state_kv,
                                                     cache->index_state_score,
                                                     DS4_N_INDEXER_HEAD_DIM,
                                                     ratio,
                                                     il,
                                                     pos,
                                                     scratch)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap,
                                   DS4_N_INDEXER_HEAD_DIM, scratch->index_comp);
            }
            if (profile) t_compress = now_sec() - t0;
        } else if (profile) {
            t_compress = now_sec() - t0;
        }
    }
    if (ratio == 4) {
        t0 = profile ? now_sec() : 0.0;
        comp_allowed = indexer_allowed_decode_one_decode_scratch(model, layer,
                                                                 scratch->attn_norm,
                                                                 scratch->qr_norm,
                                                                 cache->index_comp_kv,
                                                                 cache->n_index_comp,
                                                                 il, pos,
                                                                 scratch);
        if (profile) t_indexer = now_sec() - t0;
    }

    t0 = profile ? now_sec() : 0.0;
    if (ratio != 0) {
        layer_attention_mixed_one_decode_scratch(scratch->heads, model, layer, scratch->q,
                                                 cache->raw_kv, cache->n_raw,
                                                 cache->attn_comp_kv, cache->n_comp,
                                                 comp_allowed,
                                                 scratch);
    } else {
        layer_attention_rows_one(scratch->heads, model, layer, scratch->q, cache->raw_kv, cache->n_raw);
    }
    if (profile) t_attn_rows = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    if (profile) t_inv_rope = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_one_decode_scratch(scratch->attn_out, model, layer, scratch->heads, scratch);
    if (profile) t_out = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    hc_post_one(scratch->after_attn_hc, scratch->attn_out, scratch->attn_residual, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_ffn_one_decode_scratch(out_hc, model, layer, scratch->after_attn_hc, il, token, scratch);
    if (profile) t_ffn = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u attn hc=%.3f q=%.3f kv=%.3f rope=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f out=%.3f post=%.3f ffn=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_q * 1000.0,
                t_kv * 1000.0,
                t_rope_cache * 1000.0,
                t_compress * 1000.0,
                t_indexer * 1000.0,
                t_attn_rows * 1000.0,
                t_inv_rope * 1000.0,
                t_out * 1000.0,
                t_post * 1000.0,
                t_ffn * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

}

static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch);

/* CPU decode for one token through all 43 layers.  The caller owns scratch and
 * cache lifetimes so no per-token allocations are needed. */
static void forward_token_raw_swa_cpu_decode_scratch(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos,
        ds4_cpu_decode_scratch * scratch) {
    float *cur = scratch->cur;
    float *next = scratch->next;

    embed_token_f16(model, weights, token, scratch->plain);
    hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_raw_swa_one(next, model, &weights->layer[il], &cache->layer[il],
                                  cur, il, pos, token, scratch);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (logits) {
        output_logits_one_decode_scratch(logits, model, weights, cur, scratch);
    }
}

#ifndef DS4_NO_METAL
static void forward_token_raw_swa_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos) {
    ds4_cpu_decode_scratch scratch;
    uint32_t ctx_guess = pos + 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = cache->layer[il].compress_ratio;
        if (ratio != 0 && cache->layer[il].comp_cap > 2) {
            const uint32_t ctx_from_comp = (cache->layer[il].comp_cap - 2u) * ratio;
            if (ctx_guess < ctx_from_comp) ctx_guess = ctx_from_comp;
        }
    }
    cpu_decode_scratch_init(&scratch, ctx_guess);
    forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, cache, token, pos, &scratch);
    cpu_decode_scratch_free(&scratch);
}
#endif

/* CPU prefill in layer-major order.  All prompt tokens pass through layer 0,
 * then layer 1, etc., which exposes batch matmul opportunities. */
static void prefill_layer_major_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        const token_vec   * prompt) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t n_tok = (uint64_t)prompt->len;
    float *cur = xmalloc((size_t)n_tok * hc_dim * sizeof(cur[0]));
    float *next = xmalloc((size_t)n_tok * hc_dim * sizeof(next[0]));
    float *attn = xmalloc((size_t)n_tok * hc_dim * sizeof(attn[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    uint32_t ffn_batch = 128;
    const bool batched_attn = getenv("DS4_NO_BATCHED_ATTN") == NULL;
    const bool batched_ffn = getenv("DS4_BATCHED_FFN") != NULL;
    const bool parallel_ffn = getenv("DS4_PARALLEL_FFN") != NULL;
    const bool shared_batch_ffn = getenv("DS4_NO_SHARED_BATCH_FFN") == NULL;
    const char *batch_env = getenv("DS4_PREFILL_BATCH");
    ds4_cpu_decode_scratch decode_scratch;
    bool decode_scratch_ready = false;
    if (batch_env && batch_env[0]) {
        long v = strtol(batch_env, NULL, 10);
        if (v > 0 && v < 4096) ffn_batch = (uint32_t)v;
    }

    for (uint64_t t = 0; t < n_tok; t++) {
        embed_token_f16(model, weights, prompt->v[t], plain);
        hc_from_plain_embedding(cur + t * hc_dim, plain, DS4_N_EMBD, DS4_N_HC);
    }

    free(plain);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        fprintf(stderr, "ds4: prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
        fflush(stderr);

        if (batched_attn) {
            layer_attention_raw_swa_batch(attn,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur,
                                          (uint32_t)n_tok,
                                          il,
                                          0);

            if (batched_ffn) {
                for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                    uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                    layer_ffn_batch(next + t * hc_dim,
                                    model,
                                    &weights->layer[il],
                                    attn + t * hc_dim,
                                    prompt->v + t,
                                    nb,
                                    il);
                }
            } else if (shared_batch_ffn) {
                layer_ffn_shared_batch(next,
                                       model,
                                       &weights->layer[il],
                                       attn,
                                       prompt->v,
                                       (uint32_t)n_tok,
                                       il);
            } else if (parallel_ffn) {
                layer_ffn_tokens_parallel(next,
                                          model,
                                          &weights->layer[il],
                                          attn,
                                          prompt->v,
                                          (uint32_t)n_tok,
                                          il);
            } else {
                for (uint64_t t = 0; t < n_tok; t++) {
                    layer_ffn_one(next + t * hc_dim,
                                  model,
                                  &weights->layer[il],
                                  attn + t * hc_dim,
                                  il,
                                  prompt->v[t],
                                  false);
                }
            }
        } else if (batched_ffn) {
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_attention_raw_swa_one(attn + t * hc_dim,
                                            model,
                                            &weights->layer[il],
                                            &cache->layer[il],
                                            cur + t * hc_dim,
                                            il,
                                            (uint32_t)t);
            }

            for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                layer_ffn_batch(next + t * hc_dim,
                                model,
                                &weights->layer[il],
                                attn + t * hc_dim,
                                prompt->v + t,
                                nb,
                                il);
            }
        } else {
            if (!decode_scratch_ready) {
                cpu_decode_scratch_init(&decode_scratch, (uint32_t)n_tok);
                decode_scratch_ready = true;
            }
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_forward_raw_swa_one(next + t * hc_dim,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur + t * hc_dim,
                                          il,
                                          (uint32_t)t,
                                          prompt->v[t],
                                          &decode_scratch);
            }
        }

        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    kv_cache_finish_prefill_states(cache, (uint32_t)n_tok);

    if (logits) {
        output_logits_one(logits, model, weights, cur + (n_tok - 1) * hc_dim);
    }

    if (decode_scratch_ready) cpu_decode_scratch_free(&decode_scratch);
    free(next);
    free(cur);
    free(attn);
}

/* Diagnostic first-token layer without cache history: the token attends only
 * to itself, useful for checking a minimal end-to-end slice. */
static void layer_forward_self_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_normed_one(model, layer, attn_norm, q);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);
    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv, DS4_N_HEAD_DIM);

    layer_attention_one(heads, model, layer, q, kv);
    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    layer_ffn_one(out_hc, model, layer, after_attn_hc, il, token, false);

    free(after_attn_hc);
    free(attn_out);
    free(heads);
    free(kv);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

static void forward_first_token_cpu(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_weights * weights,
        int                 token) {
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    float *cur = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(cur[0]));
    float *next = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(next[0]));

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cur, plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_self_one(next, model, &weights->layer[il], cur, il, 0, token);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out_hc, cur, (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(out_hc[0]));

    free(next);
    free(cur);
    free(plain);
}

/* Collapse final HC streams into the ordinary embedding vector before the
 * output norm and vocabulary projection. */
static void output_hc_head_one(
        float             * out,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;
    float *flat = xmalloc((size_t)hc_dim * sizeof(flat[0]));
    float *pre = xmalloc((size_t)n_hc * sizeof(pre[0]));
    float *w = xmalloc((size_t)n_hc * sizeof(w[0]));

    rms_norm_no_weight(flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, w, DS4_N_EMBD, n_hc);

    free(w);
    free(pre);
    free(flat);
}

/* Final language-model head: HC collapse, RMSNorm, and Q8_0 vocab projection. */
static void output_logits_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    float *embd = xmalloc((size_t)DS4_N_EMBD * sizeof(embd[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    output_hc_head_one(embd, model, weights, inp_hc);
    rms_norm_weight(norm, embd, tensor_data(model, weights->output_norm), DS4_N_EMBD, DS4_RMS_EPS);

    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
    free(embd);
}

/* Allocation-free logits head for CPU decode. */
static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    rms_norm_no_weight(scratch->output_flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(scratch->output_pre, model, weights->output_hc_fn, scratch->output_flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        scratch->output_weights[i] = sigmoid_stable(scratch->output_pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(scratch->output_embd, inp_hc, scratch->output_weights, DS4_N_EMBD, n_hc);
    rms_norm_weight(scratch->output_norm, scratch->output_embd,
                    tensor_data(model, weights->output_norm),
                    DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0_decode_scratch(logits, model, weights->output, scratch->output_norm, scratch);
}

#ifndef DS4_NO_METAL
static int sample_argmax(const float *logits, uint32_t n_vocab);

/* =========================================================================
 * Metal Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the Metal executor.
 */

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}

static uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

#endif

static void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}

#ifndef DS4_NO_METAL
/* =========================================================================
 * Metal Release Graph State.
 * =========================================================================
 *
 * The release Metal executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

typedef struct {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_metal_tensor *cur_hc;
    ds4_metal_tensor *flat_hc;
    ds4_metal_tensor *hc_mix;
    ds4_metal_tensor *hc_split;
    ds4_metal_tensor *hc_pre;
    ds4_metal_tensor *hc_post;
    ds4_metal_tensor *hc_comb;
    ds4_metal_tensor *attn_cur;
    ds4_metal_tensor *attn_norm;
    ds4_metal_tensor *qr;
    ds4_metal_tensor *qr_norm;
    ds4_metal_tensor *q;
    ds4_metal_tensor *kv_raw;
    ds4_metal_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_metal_tensor *layer_raw_cache[DS4_N_LAYER];
    ds4_metal_tensor *layer_attn_comp_cache[DS4_N_LAYER];
    ds4_metal_tensor *layer_attn_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *layer_attn_state_score[DS4_N_LAYER];
    ds4_metal_tensor *layer_index_comp_cache[DS4_N_LAYER];
    ds4_metal_tensor *layer_index_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *layer_index_state_score[DS4_N_LAYER];

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  The prefix1 buffers are the cheap partial-accept state for the
     * common N=2 case. */
    ds4_metal_tensor *spec_attn_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *spec_attn_state_score[DS4_N_LAYER];
    ds4_metal_tensor *spec_index_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *spec_index_state_score[DS4_N_LAYER];
    ds4_metal_tensor *spec_prefix1_attn_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *spec_prefix1_attn_state_score[DS4_N_LAYER];
    ds4_metal_tensor *spec_prefix1_index_state_kv[DS4_N_LAYER];
    ds4_metal_tensor *spec_prefix1_index_state_score[DS4_N_LAYER];
    ds4_metal_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_N_LAYER];
    uint32_t layer_n_index_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_index_comp[DS4_N_LAYER];
    bool spec_capture_prefix1;
    uint32_t raw_cap;
    uint32_t comp_cap;

    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_metal_tensor *comp_kv_cur;
    ds4_metal_tensor *comp_sc_cur;
    ds4_metal_tensor *indexer_q;
    ds4_metal_tensor *indexer_weights;
    ds4_metal_tensor *indexer_scores;
    ds4_metal_tensor *comp_mask;
    ds4_metal_tensor *comp_selected;
    ds4_metal_tensor *heads;
    ds4_metal_tensor *attn_low;
    ds4_metal_tensor *attn_out;
    ds4_metal_tensor *after_attn_hc;
    ds4_metal_tensor *ffn_cur;
    ds4_metal_tensor *ffn_norm;
    ds4_metal_tensor *shared_gate;
    ds4_metal_tensor *shared_up;
    ds4_metal_tensor *shared_mid;
    ds4_metal_tensor *shared_out;
    ds4_metal_tensor *router_logits;
    ds4_metal_tensor *router_probs;
    ds4_metal_tensor *router_selected;
    ds4_metal_tensor *router_weights;
    ds4_metal_tensor *routed_gate;
    ds4_metal_tensor *routed_up;
    ds4_metal_tensor *routed_mid;
    ds4_metal_tensor *routed_down;
    ds4_metal_tensor *routed_out;
    ds4_metal_tensor *ffn_out;
    ds4_metal_tensor *after_ffn_hc;
    ds4_metal_tensor *output_pre;
    ds4_metal_tensor *output_weights;
    ds4_metal_tensor *output_embd;
    ds4_metal_tensor *output_norm;
    ds4_metal_tensor *logits;

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    ds4_metal_tensor *mtp_embed;
    ds4_metal_tensor *mtp_enorm;
    ds4_metal_tensor *mtp_eproj;
    ds4_metal_tensor *mtp_eproj_hc;
    ds4_metal_tensor *mtp_hnorm_hc;
    ds4_metal_tensor *mtp_hproj_hc;
    ds4_metal_tensor *mtp_input_hc;
    ds4_metal_tensor *mtp_state_hc;
    ds4_metal_tensor *mtp_next_hc;
    ds4_metal_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_metal_tensor *prefill_tokens;
    ds4_metal_tensor *batch_cur_hc;
    ds4_metal_tensor *batch_next_hc;
    ds4_metal_tensor *batch_flat_hc;
    ds4_metal_tensor *batch_hc_mix;
    ds4_metal_tensor *batch_hc_split;
    ds4_metal_tensor *batch_attn_cur;
    ds4_metal_tensor *batch_attn_norm;
    ds4_metal_tensor *batch_qr;
    ds4_metal_tensor *batch_qr_norm;
    ds4_metal_tensor *batch_q;
    ds4_metal_tensor *batch_kv_raw;
    ds4_metal_tensor *batch_kv;
    ds4_metal_tensor *batch_comp_kv;
    ds4_metal_tensor *batch_comp_sc;
    ds4_metal_tensor *batch_indexer_q;
    ds4_metal_tensor *batch_indexer_weights;
    ds4_metal_tensor *batch_heads;
    ds4_metal_tensor *batch_attn_low;
    ds4_metal_tensor *batch_attn_out;
    ds4_metal_tensor *batch_group_tmp;
    ds4_metal_tensor *batch_low_tmp;
    ds4_metal_tensor *batch_after_attn_hc;
    ds4_metal_tensor *batch_ffn_cur;
    ds4_metal_tensor *batch_ffn_norm;
    ds4_metal_tensor *batch_shared_gate;
    ds4_metal_tensor *batch_shared_up;
    ds4_metal_tensor *batch_shared_mid;
    ds4_metal_tensor *batch_shared_out;
    ds4_metal_tensor *batch_router_logits;
    ds4_metal_tensor *batch_router_probs;
    ds4_metal_tensor *batch_router_selected;
    ds4_metal_tensor *batch_router_weights;
    ds4_metal_tensor *batch_routed_gate;
    ds4_metal_tensor *batch_routed_up;
    ds4_metal_tensor *batch_routed_mid;
    ds4_metal_tensor *batch_routed_down;
    ds4_metal_tensor *batch_routed_out;
    ds4_metal_tensor *batch_ffn_out;
    bool materialize_ffn_out;
    bool quality;
    bool mtp_enabled;
} ds4_metal_graph;

/* Release every Metal tensor owned by the whole-model graph runtime. */
static void metal_graph_free(ds4_metal_graph *g) {
    ds4_metal_tensor_free(g->batch_ffn_out);
    ds4_metal_tensor_free(g->batch_routed_out);
    ds4_metal_tensor_free(g->batch_routed_down);
    ds4_metal_tensor_free(g->batch_routed_mid);
    ds4_metal_tensor_free(g->batch_routed_up);
    ds4_metal_tensor_free(g->batch_routed_gate);
    ds4_metal_tensor_free(g->batch_router_weights);
    ds4_metal_tensor_free(g->batch_router_selected);
    ds4_metal_tensor_free(g->batch_router_probs);
    ds4_metal_tensor_free(g->batch_router_logits);
    ds4_metal_tensor_free(g->batch_shared_out);
    ds4_metal_tensor_free(g->batch_shared_mid);
    ds4_metal_tensor_free(g->batch_shared_up);
    ds4_metal_tensor_free(g->batch_shared_gate);
    ds4_metal_tensor_free(g->batch_ffn_norm);
    ds4_metal_tensor_free(g->batch_ffn_cur);
    ds4_metal_tensor_free(g->batch_after_attn_hc);
    ds4_metal_tensor_free(g->batch_low_tmp);
    ds4_metal_tensor_free(g->batch_group_tmp);
    ds4_metal_tensor_free(g->batch_attn_out);
    ds4_metal_tensor_free(g->batch_attn_low);
    ds4_metal_tensor_free(g->batch_heads);
    ds4_metal_tensor_free(g->batch_indexer_weights);
    ds4_metal_tensor_free(g->batch_indexer_q);
    ds4_metal_tensor_free(g->batch_comp_sc);
    ds4_metal_tensor_free(g->batch_comp_kv);
    ds4_metal_tensor_free(g->batch_kv);
    ds4_metal_tensor_free(g->batch_kv_raw);
    ds4_metal_tensor_free(g->batch_q);
    ds4_metal_tensor_free(g->batch_qr_norm);
    ds4_metal_tensor_free(g->batch_qr);
    ds4_metal_tensor_free(g->batch_attn_norm);
    ds4_metal_tensor_free(g->batch_attn_cur);
    ds4_metal_tensor_free(g->batch_hc_split);
    ds4_metal_tensor_free(g->batch_hc_mix);
    ds4_metal_tensor_free(g->batch_flat_hc);
    ds4_metal_tensor_free(g->batch_next_hc);
    ds4_metal_tensor_free(g->batch_cur_hc);
    ds4_metal_tensor_free(g->prefill_tokens);
    ds4_metal_tensor_free(g->logits);
    ds4_metal_tensor_free(g->mtp_raw_cache);
    ds4_metal_tensor_free(g->mtp_next_hc);
    ds4_metal_tensor_free(g->mtp_state_hc);
    ds4_metal_tensor_free(g->mtp_input_hc);
    ds4_metal_tensor_free(g->mtp_hproj_hc);
    ds4_metal_tensor_free(g->mtp_hnorm_hc);
    ds4_metal_tensor_free(g->mtp_eproj_hc);
    ds4_metal_tensor_free(g->mtp_eproj);
    ds4_metal_tensor_free(g->mtp_enorm);
    ds4_metal_tensor_free(g->mtp_embed);
    ds4_metal_tensor_free(g->spec_logits);
    ds4_metal_tensor_free(g->output_norm);
    ds4_metal_tensor_free(g->output_embd);
    ds4_metal_tensor_free(g->output_weights);
    ds4_metal_tensor_free(g->output_pre);
    ds4_metal_tensor_free(g->after_ffn_hc);
    ds4_metal_tensor_free(g->ffn_out);
    ds4_metal_tensor_free(g->routed_out);
    ds4_metal_tensor_free(g->routed_down);
    ds4_metal_tensor_free(g->routed_mid);
    ds4_metal_tensor_free(g->routed_up);
    ds4_metal_tensor_free(g->routed_gate);
    ds4_metal_tensor_free(g->router_weights);
    ds4_metal_tensor_free(g->router_selected);
    ds4_metal_tensor_free(g->router_probs);
    ds4_metal_tensor_free(g->router_logits);
    ds4_metal_tensor_free(g->shared_out);
    ds4_metal_tensor_free(g->shared_mid);
    ds4_metal_tensor_free(g->shared_up);
    ds4_metal_tensor_free(g->shared_gate);
    ds4_metal_tensor_free(g->ffn_norm);
    ds4_metal_tensor_free(g->ffn_cur);
    ds4_metal_tensor_free(g->after_attn_hc);
    ds4_metal_tensor_free(g->attn_out);
    ds4_metal_tensor_free(g->attn_low);
    ds4_metal_tensor_free(g->heads);
    ds4_metal_tensor_free(g->comp_sc_cur);
    ds4_metal_tensor_free(g->comp_kv_cur);
    ds4_metal_tensor_free(g->comp_mask);
    ds4_metal_tensor_free(g->comp_selected);
    ds4_metal_tensor_free(g->indexer_scores);
    ds4_metal_tensor_free(g->indexer_weights);
    ds4_metal_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_metal_tensor_free(g->spec_attn_state_kv[il]);
        ds4_metal_tensor_free(g->spec_attn_state_score[il]);
        ds4_metal_tensor_free(g->spec_index_state_kv[il]);
        ds4_metal_tensor_free(g->spec_index_state_score[il]);
        ds4_metal_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_metal_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_metal_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_metal_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    ds4_metal_tensor_free(g->kv);
    ds4_metal_tensor_free(g->kv_raw);
    ds4_metal_tensor_free(g->q);
    ds4_metal_tensor_free(g->qr_norm);
    ds4_metal_tensor_free(g->qr);
    ds4_metal_tensor_free(g->attn_norm);
    ds4_metal_tensor_free(g->attn_cur);
    ds4_metal_tensor_free(g->hc_comb);
    ds4_metal_tensor_free(g->hc_post);
    ds4_metal_tensor_free(g->hc_pre);
    ds4_metal_tensor_free(g->hc_split);
    ds4_metal_tensor_free(g->hc_mix);
    ds4_metal_tensor_free(g->flat_hc);
    ds4_metal_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

static bool metal_tensor_fill_f32(ds4_metal_tensor *t, float v, uint64_t n) {
    float *p = ds4_metal_tensor_contents(t);
    if (!p || ds4_metal_tensor_bytes(t) < n * sizeof(float)) return false;
    for (uint64_t i = 0; i < n; i++) p[i] = v;
    return true;
}

/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return false;

    const char *name_env = getenv("DS4_METAL_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_METAL_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

static void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_metal_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_metal_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_metal_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_metal_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_metal_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_i32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_metal_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_metal_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_metal_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool metal_graph_needs_ffn_out(const ds4_metal_graph *g, uint32_t il, uint32_t pos) {
    return g->materialize_ffn_out || metal_graph_debug_wants("ffn_out", il, pos);
}

static bool metal_graph_ensure_ffn_out(ds4_metal_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}

static bool metal_graph_ensure_batch_ffn_out(ds4_metal_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_metal_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}

/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool metal_graph_alloc_raw_cap(
        ds4_metal_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp) {
    memset(g, 0, sizeof(*g));
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;

    g->cur_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_metal_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_metal_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_metal_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_metal_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_metal_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_metal_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_metal_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_metal_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_metal_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_metal_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = ds4_metal_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = ds4_metal_tensor_alloc((uint64_t)g->comp_cap * DS4_N_HEAD_DIM * sizeof(float));
            g->layer_attn_state_kv[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_mtp) {
                g->spec_attn_state_kv[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_kv[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_score[il] = ds4_metal_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = ds4_metal_tensor_alloc((uint64_t)g->comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_mtp) {
                    g->spec_index_state_kv[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_kv[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_score[il] = ds4_metal_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_metal_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_metal_tensor_alloc(comp_width_max * sizeof(float));
    g->indexer_q = ds4_metal_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_metal_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_metal_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_metal_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_metal_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_metal_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_metal_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_metal_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_metal_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_metal_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_metal_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_metal_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_metal_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_metal_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_metal_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_metal_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_metal_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_metal_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_metal_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_metal_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_metal_tensor_alloc(vocab_dim * sizeof(float));
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_metal_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_metal_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = ds4_metal_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->spec_logits = ds4_metal_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
        g->mtp_n_raw = 0;
    }

    g->prefill_tokens = ds4_metal_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_metal_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_metal_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_metal_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_metal_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_metal_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_metal_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_metal_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_metal_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_metal_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_metal_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_metal_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_metal_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_metal_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_metal_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_metal_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_metal_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_metal_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_metal_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_metal_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_metal_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_metal_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_metal_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_metal_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_metal_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL &&
                               g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL &&
                               g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache && g->spec_logits)) &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) metal_graph_free(g);
    return ok;
}

static bool metal_graph_alloc(
        ds4_metal_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false);
}

static uint32_t metal_graph_raw_span_for_batch(
        const ds4_metal_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t metal_graph_raw_start_for_span(
        const ds4_metal_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool metal_graph_capture_prefix1_attn_state(ds4_metal_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_metal_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_metal_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_metal_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix1_index_state(ds4_metal_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_metal_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_metal_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_metal_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

static uint32_t metal_graph_decode_indexer_top_k(const ds4_metal_graph *g) {
    (void)g;
    return DS4_N_INDEXER_TOP_K;
}

/* =========================================================================
 * Metal Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_METAL_DISABLE_*_FUSION environment switches.
 */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

static bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool metal_graph_decode_hc_pre(
        ds4_metal_tensor       *out,
        ds4_metal_tensor       *split,
        const ds4_metal_tensor *mix,
        const ds4_metal_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (metal_graph_use_reference_hc_decode()) {
        return ds4_metal_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_metal_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_metal_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

static bool metal_graph_decode_kv_store(
        ds4_metal_tensor *kv,
        ds4_metal_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (metal_graph_use_reference_kv_decode()) {
        return ds4_metal_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_metal_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_metal_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool metal_graph_matmul_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok);

static bool metal_graph_encode_decode_layer(
        ds4_metal_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_metal_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_metal_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_metal_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_metal_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->qr, model->map, model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD, q_rank,
                                              g->attn_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_metal_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_metal_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_metal_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qnorm", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_metal_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_metal_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    if (ok) ok = ds4_metal_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = metal_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    DS4_METAL_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    ds4_metal_tensor *comp_cache = NULL;
    ds4_metal_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->comp_cap) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
            ok = ds4_metal_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                  g->comp_sc_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_compressor_kv->abs_offset,
                                                  layer->attn_compressor_gate->abs_offset,
                                                  DS4_N_EMBD,
                                                  comp_width,
                                                  g->attn_norm,
                                                  1) != 0;
        } else {
            if (ok) ok = ds4_metal_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
            if (ok) ok = ds4_metal_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_metal_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        g->layer_attn_comp_cache[il],
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        comp_row,
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS) != 0;
        if (ok && emit) {
            ds4_metal_tensor *comp_row_view = ds4_metal_tensor_view(
                    g->layer_attn_comp_cache[il],
                    (uint64_t)comp_row * DS4_N_HEAD_DIM * sizeof(float),
                    (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = ds4_metal_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    metal_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
                ds4_metal_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->comp_cap) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
                ok = ds4_metal_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->indexer_compressor_kv->abs_offset,
                                                      layer->indexer_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      index_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                if (ok) ok = ds4_metal_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
                if (ok) ok = ds4_metal_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_metal_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_top_k = metal_graph_decode_indexer_top_k(g);
            if (ok && g->layer_n_comp[il] > decode_top_k) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = ds4_metal_matmul_f16_tensor(g->indexer_q, model->map, model->size,
                                                         layer->indexer_attn_q_b->abs_offset,
                                                         q_rank, indexer_q_dim,
                                                         g->qr_norm, 1) != 0;
                if (ok) ok = ds4_metal_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_metal_matmul_f16_tensor(g->indexer_weights, model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         g->attn_norm, 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_metal_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_metal_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           decode_top_k) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    n_selected = decode_top_k < g->layer_n_index_comp[il]
                        ? decode_top_k
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_METAL_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_metal_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    comp_cache,
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_metal_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_metal_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc = !metal_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_metal_attention_output_low_q8_tensor(g->attn_low,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      g->heads) != 0;
        if (ok) {
            ok = ds4_metal_matmul_q8_0_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_metal_attention_output_q8_batch_tensor(g->attn_out,
                                                        g->attn_low,
                                                        g->batch_group_tmp,
                                                        g->batch_low_tmp,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        g->heads, 1) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_metal_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = ds4_metal_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_metal_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_metal_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = ds4_metal_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    if (ok) ok = ds4_metal_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 model->map, model->size,
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    const bool fuse_shared_gate_up =
        !g->quality &&
        getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    if (ok && fuse_shared_gate_up) {
        ok = ds4_metal_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm) != 0;
    } else {
        if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_metal_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up, shared_dim, 0.0f, 1.0f) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_down_hc =
        !keep_ffn_out && !metal_graph_use_reference_shared_down_hc();
    if (ok && fuse_shared_down_hc) {
        ok = ds4_metal_shared_down_hc_expand_q8_0_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_metal_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_metal_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_shared_down_hc) {
        ok = ds4_metal_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
static bool metal_graph_encode_output_head(
        ds4_metal_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_metal_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_metal_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_metal_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_metal_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_metal_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
static bool metal_graph_encode_output_head_batch(
        ds4_metal_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_metal_tensor *output_pre = NULL;
    ds4_metal_tensor *output_weights = NULL;
    ds4_metal_tensor *output_embd = NULL;
    ds4_metal_tensor *output_norm = NULL;
    ds4_metal_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_metal_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_metal_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_metal_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_metal_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_metal_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_metal_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_metal_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_metal_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_metal_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_metal_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_metal_matmul_q8_0_tensor(logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_metal_tensor_free(logits);
    ds4_metal_tensor_free(output_norm);
    ds4_metal_tensor_free(output_embd);
    ds4_metal_tensor_free(output_weights);
    ds4_metal_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_matmul_plain_tensor(
        ds4_metal_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_metal_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_metal_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_metal_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

static bool metal_graph_encode_output_head_mtp(
        ds4_metal_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_metal_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->output_pre, mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, g->flat_hc, 1);
    if (ok) ok = ds4_metal_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_metal_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_metal_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_metal_matmul_q8_0_tensor(g->logits,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    return ok;
}

/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void metal_graph_trace_layer_stages(
        ds4_metal_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];

    bool ok = ds4_metal_tensor_read(g->attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->shared_gate, 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->shared_up, 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->shared_mid, 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_metal_tensor_read(g->router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_metal_tensor_read(g->routed_mid, 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_metal_tensor_read(g->cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

static int metal_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_metal_graph g;
    bool ok = metal_graph_alloc(&g, weights, layer);
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_metal_begin_commands() != 0;
    if (ok) ok = ds4_metal_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = metal_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_metal_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_metal_end_commands() != 0;

    if (ok) {
        ok = ds4_metal_tensor_read(g.after_ffn_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_metal_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_metal_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_metal_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_metal_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}
