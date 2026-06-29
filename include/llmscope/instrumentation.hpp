#pragma once
#include <atomic>
#include <mutex>
#include <vector>
#include "event.hpp"
#include "tensor.hpp"

namespace llmscope {

uint64_t now_ns();

class HookRegistry {
public:
    void attach(TraceSink* sink);
    void detach(TraceSink* sink);
    bool active() const;

    void emit_tensor(const HookPoint& point,
                     const Tensor& output,
                     double latency_ms);

    void emit_attention(const HookPoint& point,
                        const Tensor& output,
                        double latency_ms,
                        int num_heads,
                        int seq_len,
                        std::vector<std::vector<float>> attention,
                        std::vector<std::string> tokens);

private:
    mutable std::mutex mutex_;
    std::vector<TraceSink*> sinks_;
    std::atomic<uint64_t> next_id_{1};
    float zero_eps_ = 1e-6f;

    void dispatch(const TraceEvent& ev);
};

}  // namespace llmscope