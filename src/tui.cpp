#include <iostream>
#include "../include/tui.hpp"

using namespace std;

void renderUI(RingBuffer& rb) {
    cout << "=============================================\n";
    cout << "          TRACEFORMER - TUI SKELETON         \n";
    cout << "=============================================\n";

    cout << "\n1. MODEL TOPOLOGY\n";
    cout << "---------------------------------------------\n";
    cout << "llama-3-8b\n";
    cout << " |- embed_tokens\n";
    cout << " |- layers.0\n";
    cout << " |- layers.1\n";

    cout << "\n2. LIVE PACKET STREAM\n";
    cout << "---------------------------------------------\n";

    auto data = rb.getAll();

    for (auto &m : data) {
        cout << "Token: " << m.token_id
             << " | Layer: " << m.layer_name
             << " | Latency: " << m.latency_ms
             << " ms\n";
    }

    cout << "\n3. RUNTIME METRICS\n";
    cout << "---------------------------------------------\n";
    cout << "Tensor Shape: [1, 32, 4096]\n";
    cout << "Sparsity Rate: 54.2%\n";
}