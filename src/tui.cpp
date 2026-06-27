#include <iostream>
#include "../include/tui.hpp"

void renderUI(const RingBuffer& rb) {
    std::cout << "TRACEFORMER\n";
    std::cout << "Events stored: " << rb.size() << "/" << rb.getCapacity() << "\n";

    auto events = rb.getAll();
    for (const auto& e : events) {
        std::cout << "Event " << e.event_id
                  << " | Token " << e.token_id
                  << " | " << e.layer_name
                  << " | " << e.submodule_name
                  << " | latency=" << e.latency_ms << "ms"
                  << " | anomaly=" << (e.anomaly_flag ? "YES" : "NO")
                  << "\n";
    }
}