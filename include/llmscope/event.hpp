#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tensor.hpp"

namespace llmscope {

enum class ModuleKind {
    Embedding,
    LayerNorm,
    Attention,
    AttentionScores,
    MLP,
    Residual,
    LogitsHead,
    Other
};

struct HookPoint {
    std::string module_path;
    ModuleKind kind = ModuleKind::Other;
    int layer_index = -1;
    int token_step = 0;
};

struct TraceEvent {
    uint64_t id = 0;
    uint64_t timestamp_ns = 0;
    int token_step = 0;
    int layer_index = -1;
    ModuleKind kind = ModuleKind::Other;
    std::string module_path;
    TensorStats stats;
    double latency_ms = 0.0;

    int num_heads = 0;
    int seq_len = 0;
    std::vector<std::vector<float>> attention;
    std::vector<std::string> tokens;

    bool has_attention() const {
        return !attention.empty() && !tokens.empty();
    }
};

class TraceSink {
public:
    virtual ~TraceSink() = default;
    virtual void on_event(const TraceEvent& ev) = 0;
};

}  // namespace llmscope