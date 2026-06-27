#include "../include/dashboard.hpp"
#include <iostream>

void runDashboard(const RingBuffer& rb) {
    std::cout << "[Tab]: Cycle Focus  |  [Q]: Quit App\n\n";

    std::cout << "+------------------------------ 1. MODEL TOPOLOGY ------------------------------+\n";
    std::cout << "| > llama-3-8b                                                                  |\n";
    std::cout << "|   > embed_tokens                                                              |\n";
    std::cout << "|   > layers                                                                    |\n";
    std::cout << "|     > layers.0                                                                |\n";
    std::cout << "|     > layers.1  [Active Capture Target]                                       |\n";
    std::cout << "|       - layers.1.attn                                                         |\n";
    std::cout << "|       - layers.1.mlp                                                          |\n";
    std::cout << "|     > layers.2                                                                |\n";
    std::cout << "+------------------------------------------------------------------------------+\n\n";

    std::cout << "+----------------------------- 2. LIVE PACKET STREAM --------------------------+\n";
    std::cout << "| ID   | TIMESTAMP   | LAYER TYPE    | COMPUTE DEVICE                          |\n";
    std::cout << "+------+-------------+---------------+----------------------------------------+\n";

    auto events = rb.getAll();
    for (const auto& e : events) {
        std::string layerType = (e.submodule_name == "attn") ? "Attn (Self)" : "MLP (SwiGLU)";
        std::string device = e.anomaly_flag ? "CPU (Fallback)" : "CUDA [GPU 0]";

        std::cout << "| " << e.event_id
                  << "    | " << e.timestamp_ms
                  << " | " << layerType
                  << " | " << device << "\n";
    }

    std::cout << "+------------------------------------------------------------------------------+\n\n";

    std::cout << "+--------------------- 3. ATTENTION MATRIX VISUALIZER ------------------------+\n";
    std::cout << "| Tokens: [I] [want] [it] [to] [be] [keyboard] [driven]                       |\n";
    std::cout << "| [I]       ##  ..  ..  ..  ..  ..  ..                                       |\n";
    std::cout << "| [want]    ..  ##  ..  ..  ..  ..  ..                                       |\n";
    std::cout << "| [it]      ..  ..  ##  ..  ..  ..  ..                                       |\n";
    std::cout << "| [to]      ..  ..  ..  ##  ..  ..  ..                                       |\n";
    std::cout << "+------------------------------------------------------------------------------+\n\n";

    std::cout << "+---------------------- 4. RUNTIME METRICS INSPECTOR -------------------------+\n";
    if (!events.empty()) {
        const auto& e = events.back();
        std::cout << "| Tensor Shape : " << e.tensor_shape << "   Dtype: " << e.dtype << "\n";
        std::cout << "| Sparsity Rate : " << e.sparsity_rate * 100.0 << "%\n";
        std::cout << "| Latency Delta : " << e.latency_ms << " ms\n";
    }
    std::cout << "+------------------------------------------------------------------------------+\n\n";

    std::cout << "+---------------------- 5. NUMERICAL ANOMALY LEDGER --------------------------+\n";
    for (const auto& e : events) {
        if (e.anomaly_flag) {
            std::cout << "| " << e.timestamp_ms << "  ALERT  " << e.layer_name << " / " << e.submodule_name << "\n";
        }
    }
    std::cout << "+------------------------------------------------------------------------------+\n";
}