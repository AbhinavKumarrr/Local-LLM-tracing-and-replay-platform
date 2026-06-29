#include "llmscope/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

namespace llmscope {

namespace {
inline float gelu(float x) {
    constexpr float kC = 0.7978845608028654f;
    const float inner = kC * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}
}  // namespace

Transformer::Transformer(TransformerConfig cfg, HookRegistry* hooks)
    : cfg_(std::move(cfg)), hooks_(hooks) {
    layers_.reserve(static_cast<std::size_t>(cfg_.n_layers));
    for (int l = 0; l < cfg_.n_layers; ++l) {
        const unsigned base = cfg_.seed + static_cast<unsigned>(l) * 100u;
        LayerWeights lw;
        lw.wq = make_matrix(cfg_.d_model, cfg_.d_model, base + 1u);
        lw.wk = make_matrix(cfg_.d_model, cfg_.d_model, base + 2u);
        lw.wv = make_matrix(cfg_.d_model, cfg_.d_model, base + 3u);
        lw.wo = make_matrix(cfg_.d_model, cfg_.d_model, base + 4u);
        lw.w1 = make_matrix(cfg_.d_model, cfg_.d_ff, base + 5u);
        lw.w2 = make_matrix(cfg_.d_ff, cfg_.d_model, base + 6u);
        layers_.push_back(std::move(lw));
    }
}

Transformer::Matrix Transformer::make_matrix(int rows, int cols, unsigned seed) const {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.w.resize(static_cast<std::size_t>(rows) * cols);
    std::mt19937 rng(seed);
    const float scale = 1.0f / std::sqrt(static_cast<float>(rows));
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& v : m.w) v = dist(rng);
    return m;
}

Tensor Transformer::embed(const std::vector<int>& tokens) const {
    const int seq = static_cast<int>(tokens.size());
    Tensor t({seq, cfg_.d_model});
    for (int s = 0; s < seq; ++s) {
        std::mt19937 rng(static_cast<unsigned>(tokens[s]) * 2654435761u + 1u);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int d = 0; d < cfg_.d_model; ++d) {
            t.data[static_cast<std::size_t>(s) * cfg_.d_model + d] = dist(rng);
        }
    }
    return t;
}

std::vector<float> Transformer::matmul(const std::vector<float>& x, int seq,
                                       const Matrix& w) const {
    std::vector<float> out(static_cast<std::size_t>(seq) * w.cols, 0.0f);
    for (int s = 0; s < seq; ++s) {
        const float* xr = &x[static_cast<std::size_t>(s) * w.rows];
        float* orow = &out[static_cast<std::size_t>(s) * w.cols];
        for (int k = 0; k < w.rows; ++k) {
            const float xv = xr[k];
            if (xv == 0.0f) continue;
            const float* wr = &w.w[static_cast<std::size_t>(k) * w.cols];
            for (int c = 0; c < w.cols; ++c) {
                orow[c] += xv * wr[c];
            }
        }
    }
    return out;
}

void Transformer::emit(ModuleKind kind,
                       const std::string& path,
                       int layer,
                       int step,
                       const std::vector<float>& data,
                       int seq,
                       int dim,
                       double latency_ms) {
    if (!hooks_) return;
    Tensor t;
    t.shape = {seq, dim};
    t.data = data;
    HookPoint hp;
    hp.module_path = path;
    hp.kind = kind;
    hp.layer_index = layer;
    hp.token_step = step;
    hooks_->emit_tensor(hp, t, latency_ms);
}

void Transformer::forward(const std::vector<int>& tokens,
                          const std::vector<std::string>& token_strs,
                          int token_step) {
    const int seq = static_cast<int>(tokens.size());
    if (seq == 0) return;

    const int d = cfg_.d_model;
    const int h = std::max(1, cfg_.n_heads);
    const int dh = std::max(1, d / h);

    ScopedTimer timer;

    Tensor emb = embed(tokens);
    std::vector<float> x = emb.data;
    emit(ModuleKind::Embedding, "embed_tokens", -1, token_step, x, seq, d,
         timer.elapsed_ms());

    auto layernorm = [&](const std::vector<float>& in) {
        std::vector<float> out(in.size());
        for (int s = 0; s < seq; ++s) {
            const float* row = &in[static_cast<std::size_t>(s) * d];
            float mean = 0.0f;
            for (int j = 0; j < d; ++j) mean += row[j];
            mean /= static_cast<float>(d);
            float var = 0.0f;
            for (int j = 0; j < d; ++j) {
                const float c = row[j] - mean;
                var += c * c;
            }
            var /= static_cast<float>(d);
            const float inv = 1.0f / std::sqrt(var + 1e-5f);
            float* o = &out[static_cast<std::size_t>(s) * d];
            for (int j = 0; j < d; ++j) o[j] = (row[j] - mean) * inv;
        }
        return out;
    };

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const LayerWeights& lw = layers_[static_cast<std::size_t>(l)];
        const std::string base = "layers." + std::to_string(l);

        timer.reset();
        std::vector<float> ln1 = layernorm(x);
        emit(ModuleKind::LayerNorm, base + ".input_layernorm", l, token_step,
             ln1, seq, d, timer.elapsed_ms());

        timer.reset();
        std::vector<float> Q = matmul(ln1, seq, lw.wq);
        std::vector<float> K = matmul(ln1, seq, lw.wk);
        std::vector<float> V = matmul(ln1, seq, lw.wv);

        std::vector<float> attn_out(static_cast<std::size_t>(seq) * d, 0.0f);
        std::vector<std::vector<float>> attn_matrices(static_cast<std::size_t>(h));
        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        for (int head = 0; head < h; ++head) {
            const int off = head * dh;
            std::vector<float> matrix(static_cast<std::size_t>(seq) * seq, 0.0f);

            for (int qi = 0; qi < seq; ++qi) {
                const float* qrow = &Q[static_cast<std::size_t>(qi) * d + off];
                std::vector<float> scores(seq, 0.0f);
                float maxs = -std::numeric_limits<float>::infinity();
                const int last = cfg_.causal ? qi : seq - 1;

                for (int ki = 0; ki <= last; ++ki) {
                    const float* krow = &K[static_cast<std::size_t>(ki) * d + off];
                    float dot = 0.0f;
                    for (int e = 0; e < dh; ++e) dot += qrow[e] * krow[e];
                    dot *= scale;
                    scores[ki] = dot;
                    if (dot > maxs) maxs = dot;
                }

                float denom = 0.0f;
                for (int ki = 0; ki <= last; ++ki) {
                    scores[ki] = std::exp(scores[ki] - maxs);
                    denom += scores[ki];
                }
                if (denom <= 0.0f) denom = 1.0f;

                float* mrow = &matrix[static_cast<std::size_t>(qi) * seq];
                float* outrow = &attn_out[static_cast<std::size_t>(qi) * d + off];

                for (int ki = 0; ki <= last; ++ki) {
                    const float p = scores[ki] / denom;
                    mrow[ki] = p;
                    const float* vrow = &V[static_cast<std::size_t>(ki) * d + off];
                    for (int e = 0; e < dh; ++e) outrow[e] += p * vrow[e];
                }
            }

            attn_matrices[static_cast<std::size_t>(head)] = std::move(matrix);
        }

        std::vector<float> proj = matmul(attn_out, seq, lw.wo);
        const double attn_latency = timer.elapsed_ms();

        if (hooks_) {
            Tensor t;
            t.shape = {seq, d};
            t.data = proj;
            HookPoint hp;
            hp.module_path = base + ".attn";
            hp.kind = ModuleKind::Attention;
            hp.layer_index = l;
            hp.token_step = token_step;
            hooks_->emit_attention(hp, t, attn_latency, h, seq, attn_matrices, token_strs);
        }

        for (std::size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

        timer.reset();
        std::vector<float> ln2 = layernorm(x);
        emit(ModuleKind::LayerNorm, base + ".post_attention_layernorm", l, token_step,
             ln2, seq, d, timer.elapsed_ms());

        timer.reset();
        std::vector<float> hidden = matmul(ln2, seq, lw.w1);
        for (float& v : hidden) v = gelu(v);
        std::vector<float> mlp_out = matmul(hidden, seq, lw.w2);
        emit(ModuleKind::MLP, base + ".mlp", l, token_step,
             mlp_out, seq, d, timer.elapsed_ms());

        for (std::size_t i = 0; i < x.size(); ++i) x[i] += mlp_out[i];
    }

    timer.reset();
    std::vector<float> final_norm = layernorm(x);
    emit(ModuleKind::LayerNorm, "norm", -1, token_step, final_norm, seq, d,
         timer.elapsed_ms());

    emit(ModuleKind::LogitsHead, "lm_head", -1, token_step, final_norm, seq, d, 0.0);
}

}  // namespace llmscope