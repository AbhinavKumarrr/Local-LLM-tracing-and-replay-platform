#include "../include/dashboard.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#ifdef _WIN32
#include <conio.h>
#endif

static void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

static void drawLine(char c, int n) {
    for (int i = 0; i < n; ++i) std::cout << c;
    std::cout << '\n';
}

static void renderTopology(int focusLayer) {
    std::cout << "╔══ 1. MODEL TOPOLOGY (Focus Active) ═════════════════════════════╗\n";
    std::cout << "║ ▼ llama-3-8b                                                    ║\n";
    std::cout << "║   ► embed_tokens                                                ║\n";
    std::cout << "║   ▼ layers                                                      ║\n";
    std::cout << "║    ▶ layers.0                                                   ║\n";
    if (focusLayer == 1) {
        std::cout << "║    ▼ layers.1  [Active Capture Target]                          ║\n";
        std::cout << "║      ● layers.1.attn                                            ║\n";
        std::cout << "║      ● layers.1.mlp                                             ║\n";
    } else {
        std::cout << "║    ► layers.1                                                   ║\n";
        std::cout << "║      ● layers.1.attn                                            ║\n";
        std::cout << "║      ● layers.1.mlp                                             ║\n";
    }
    std::cout << "║    ► layers.2                                                   ║\n";
    std::cout << "╚════════════════════════════ [j/k navigate] ══════════════════════╝\n";
}

static void renderStream(const RingBuffer& rb) {
    std::cout << "┌── 2. LIVE PACKET STREAM ────────────────────────────────────────┐\n";
    std::cout << "│  ID  │ TIMESTAMP    │ LAYER TYPE   │ COMPUTE DEVICE           │\n";
    std::cout << "├──────┼──────────────┼──────────────┼──────────────────────────┤\n";

    auto events = rb.getAll();
    for (const auto& e : events) {
        std::string layerType = (e.submodule_name == "attn") ? "Attn (Self)" : "MLP (SwiGLU)";
        std::string device = e.anomaly_flag ? "CPU (Fallback)" : "CUDA [GPU 0]";

        std::cout << "│ "
                  << (e.event_id < 10 ? " " : "") << e.event_id
                  << "   │ ";

        std::cout.width(12);
        std::cout << std::left << e.timestamp_ms << " │ ";

        std::cout.width(12);
        std::cout << std::left << layerType << " │ ";

        std::cout.width(22);
        std::cout << std::left << device << "│\n";
    }

    std::cout << "└────────────────────────────────────────────────────────────────┘\n";
}

static void renderAttentionPanel() {
    std::cout << "┌── 3. ATTENTION MATRIX VISUALIZER (HEAD 0) ─────────────────────┐\n";
    std::cout << "│ Tokens: [I] [want] [it] [to] [be] [keyboard] [driven]          │\n";
    std::cout << "│ [I]       ██   ░░   ░░   ░░   ░░   ░░   ░░                     │\n";
    std::cout << "│ [want]    ▒▒   ██   ░░   ░░   ░░   ░░   ░░                     │\n";
    std::cout << "│ [it]      ░░   ▒▒   ██   ░░   ░░   ░░   ░░                     │\n";
    std::cout << "│ [to]      ░░   ░░   ▒▒   ██   ░░   ░░   ░░                     │\n";
    std::cout << "│ [be]      ░░   ░░   ░░   ▒▒   ██   ░░   ░░                     │\n";
    std::cout << "│ [keyboard]░░   ░░   ░░   ░░   ▒▒   ██   ░░                     │\n";
    std::cout << "│ [driven]  ░░   ░░   ░░   ░░   ░░   ▒▒   ██                     │\n";
    std::cout << "│ Viewport Window: [0-7] x [0-7]                                  │\n";
    std::cout << "│ [Tab]: Focus | [Arrows/hjkl]: Pan | [+/-]: Contrast             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────┘\n";
}

static void renderMetrics(const Metrics& e) {
    std::cout << "┌── 4. RUNTIME METRICS INSPECTOR ─────────────────────┐\n";
    std::cout << "│ Tensor Shape : " << e.tensor_shape << "   Dtype: " << e.dtype << "\n";
    std::cout << "│ Sparsity Rate: ";
    int filled = static_cast<int>(e.sparsity_rate * 20.0);
    for (int i = 0; i < 20; ++i) std::cout << (i < filled ? '█' : '░');
    std::cout << " " << (e.sparsity_rate * 100.0) << "%\n";
    std::cout << "│ Latency Delta: " << e.latency_ms << " ms (Within Normal Bounds)\n";
    std::cout << "└──────────────────────────────────────────────────────┘\n";
}

static void renderAnomalies(const RingBuffer& rb) {
    std::cout << "┌── 5. NUMERICAL ANOMALY LEDGER ───────────────────────┐\n";
    auto events = rb.getAll();
    for (const auto& e : events) {
        if (e.anomaly_flag) {
            std::cout << "│ " << e.timestamp_ms << " ⚠ Outlier Feature: "
                      << e.layer_name << " / " << e.submodule_name
                      << "                                      │\n";
        }
    }
    std::cout << "└──────────────────────────────────────────────────────┘\n";
}

void runDashboard(const RingBuffer& rb) {
    int focus = 0;
    int selectedLayer = 1;

    while (true) {
        clearScreen();

        std::cout << "[Tab]: Cycle Focus  |  [Q]: Quit App\n";

        renderTopology(selectedLayer);
        renderStream(rb);
        renderAttentionPanel();

        auto events = rb.getAll();
        if (!events.empty()) {
            renderMetrics(events.back());
        }

        renderAnomalies(rb);

        std::cout << "\nFocus = " << focus << " | Selected layer = " << selectedLayer << "\n";
        std::cout << "Press key: ";

#ifdef _WIN32
        int ch = _getch();
        if (ch == 9) {
            focus = (focus + 1) % 5;
        } else if (ch == 'q' || ch == 'Q') {
            break;
        } else if (ch == 'j' && selectedLayer < 2) {
            ++selectedLayer;
        } else if (ch == 'k' && selectedLayer > 0) {
            --selectedLayer;
        }
#else
        char ch;
        std::cin >> ch;
        if (ch == 'q' || ch == 'Q') break;
        if (ch == '\t') focus = (focus + 1) % 5;
        if (ch == 'j' && selectedLayer < 2) ++selectedLayer;
        if (ch == 'k' && selectedLayer > 0) --selectedLayer;
#endif
    }
}