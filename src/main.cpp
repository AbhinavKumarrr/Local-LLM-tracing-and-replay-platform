#include <iostream>
#include "../include/ring_buffer.hpp"

int main() {
    RingBuffer rb(5);

    rb.push({1, 101, "layers.0", "attn", "[1, 32, 4096]", "float16", 21.114, 1.142, 0.542, 0.12, 1.45, false});
    rb.push({2, 102, "layers.0", "mlp", "[1, 32, 4096]", "float16", 21.118, 1.380, 0.538, 0.14, 1.62, false});
    rb.push({3, 103, "layers.1", "attn", "[1, 32, 4096]", "float16", 21.122, 1.510, 0.551, 0.11, 1.71, false});
    rb.push({4, 104, "layers.1", "mlp", "[1, 32, 4096]", "float16", 21.128, 1.230, 0.547, 0.10, 1.54, false});
    rb.push({5, 105, "layers.2", "attn", "[1, 32, 4096]", "float16", 21.133, 1.660, 0.560, 0.09, 1.80, true});

    auto events = rb.getAll();

    for (const auto& e : events) {
        std::cout << "Event " << e.event_id
                  << " | Token " << e.token_id
                  << " | " << e.layer_name
                  << " | " << e.submodule_name
                  << " | latency=" << e.latency_ms
                  << "ms"
                  << " | sparsity=" << e.sparsity_rate
                  << " | anomaly=" << (e.anomaly_flag ? "YES" : "NO")
                  << "\n";
    }

    return 0;
}