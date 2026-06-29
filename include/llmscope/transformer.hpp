#pragma once
#include <string>
#include <vector>
#include "event.hpp"
#include "instrumentation.hpp"
#include "tensor.hpp"

namespace llmscope {

struct TransformerConfig {
    std::string model_name = "llama-3-8b";
    int n_layers = 8;
    int n_heads = 8;
    int d_model = 128;
    int d_ff = 512;
    bool causal = true;
    unsigned seed = 1234;
};

class Transformer {
public:
    Transformer(TransformerConfig cfg, HookRegistry* hooks);
    void forward(const std::vector<int>& tokens,
                 const std::vector<std::string>& token_strs,
                 int token_step);

private:
    struct Matrix {
        int rows = 0;
        int cols = 0;
        std::vector<float> w;
    };

    struct LayerWeights {
        Matrix wq, wk, wv, wo, w1, w2;
    };

    TransformerConfig cfg_;
    HookRegistry* hooks_ = nullptr;
    std::vector<LayerWeights> layers_;

    Matrix make_matrix(int rows, int cols, unsigned seed) const;
    Tensor embed(const std::vector<int>& tokens) const;
    std::vector<float> matmul(const std::vector<float>& x, int seq, const Matrix& w) const;

    void emit(ModuleKind kind,
              const std::string& path,
              int layer,
              int step,
              const std::vector<float>& data,
              int seq,
              int dim,
              double latency_ms);
};

}  // namespace llmscope