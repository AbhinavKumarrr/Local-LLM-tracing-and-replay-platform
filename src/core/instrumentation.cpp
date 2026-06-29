#include "llmscope/instrumentation.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace llmscope {

uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

void HookRegistry::attach(TraceSink* sink) {
    if (!sink) return;
    std::lock_guard<std::mutex> lk(mutex_);
    if (std::find(sinks_.begin(), sinks_.end(), sink) == sinks_.end()) {
        sinks_.push_back(sink);
    }
}

void HookRegistry::detach(TraceSink* sink) {
    std::lock_guard<std::mutex> lk(mutex_);
    sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
}

bool HookRegistry::active() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return !sinks_.empty();
}

void HookRegistry::dispatch(const TraceEvent& ev) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (TraceSink* s : sinks_) {
        s->on_event(ev);
    }
}

void HookRegistry::emit_tensor(const HookPoint& point,
                               const Tensor& output,
                               double latency_ms) {
    TraceEvent ev;
    ev.id = next_id_.fetch_add(1);
    ev.timestamp_ns = now_ns();
    ev.token_step = point.token_step;
    ev.layer_index = point.layer_index;
    ev.kind = point.kind;
    ev.module_path = point.module_path;
    ev.stats = TensorStats::compute(output, zero_eps_);
    ev.latency_ms = latency_ms;
    dispatch(ev);
}

void HookRegistry::emit_attention(const HookPoint& point,
                                  const Tensor& output,
                                  double latency_ms,
                                  int num_heads,
                                  int seq_len,
                                  std::vector<std::vector<float>> attention,
                                  std::vector<std::string> tokens) {
    TraceEvent ev;
    ev.id = next_id_.fetch_add(1);
    ev.timestamp_ns = now_ns();
    ev.token_step = point.token_step;
    ev.layer_index = point.layer_index;
    ev.kind = point.kind;
    ev.module_path = point.module_path;
    ev.stats = TensorStats::compute(output, zero_eps_);
    ev.latency_ms = latency_ms;
    ev.num_heads = num_heads;
    ev.seq_len = seq_len;
    ev.attention = std::move(attention);
    ev.tokens = std::move(tokens);
    dispatch(ev);
}

}  // namespace llmscope