#include "../include/dashboard.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

static std::string repeatChar(char c, std::size_t n) {
    return std::string(n, c);
}

static std::string fitText(const std::string& s, std::size_t width) {
    if (s.size() >= width) return s.substr(0, width);
    return s + std::string(width - s.size(), ' ');
}

static std::string toFixed3(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << v;
    return oss.str();
}

static std::string layerType(const std::string& submodule) {
    if (submodule == "attn") return "Attn (Self)";
    if (submodule == "mlp") return "MLP (SwiGLU)";
    return submodule;
}

static std::string computeDevice(bool anomaly) {
    return anomaly ? "CPU (Fallback)" : "CUDA [GPU 0]";
}

static bool matchesLayer(const Metrics& e, int selectedLayer) {
    std::string target = "layers." + std::to_string(selectedLayer);
    return e.layer_name.find(target) != std::string::npos;
}

static const Metrics* pickSelectedMetric(const std::vector<Metrics>& events, int selectedLayer) {
    for (const auto& e : events) {
        if (matchesLayer(e, selectedLayer)) return &e;
    }
    if (!events.empty()) return &events.back();
    return nullptr;
}

static void renderHelpBar(int focus) {
    static const char* names[] = {
        "1. MODEL TOPOLOGY",
        "2. LIVE PACKET STREAM",
        "3. ATTENTION MATRIX",
        "4. RUNTIME METRICS",
        "5. ANOMALY LEDGER"
    };

    std::cout << "[Tab]: Cycle Focus  |  [j/k]: Select Layer  |  [Q]: Quit App\n";
    std::cout << "Focus: " << names[focus] << "\n\n";
}

static void renderTopology(int focus, int selectedLayer) {
    std::cout << "+------------------------- 1. MODEL TOPOLOGY "
              << (focus == 0 ? "[FOCUS ACTIVE]" : "")
              << " -------------------------+\n";

    std::cout << "| > llama-3-8b                                                     |\n";
    std::cout << "|   > embed_tokens                                                  |\n";
    std::cout << "|   > layers                                                        |\n";

    for (int i = 0; i < 3; ++i) {
        bool selected = (i == selectedLayer);
        std::string prefix = selected ? "> " : "  ";

        std::cout << "|   " << prefix << "layers." << i;
        if (selected) std::cout << "  [Active Capture Target]";
        std::cout << std::string(63 - (selected ? 0 : 2) - std::min<std::size_t>(50, 10 + (selected ? 24 : 0)), ' ')
                  << "|\n";
    }

    std::cout << "+-------------------------------------------------------------------+\n";
}

static void renderStream(const std::vector<Metrics>& events, int focus) {
    std::cout << "+---------------------- 2. LIVE PACKET STREAM "
              << (focus == 1 ? "[FOCUS ACTIVE]" : "")
              << " ----------------------+\n";
    std::cout << "| ID   | TIMESTAMP   | LAYER TYPE     | COMPUTE DEVICE              |\n";
    std::cout << "+------+-------------+----------------+-----------------------------+\n";

    for (const auto& e : events) {
        std::string id = fitText(std::to_string(e.event_id), 4);
        std::string ts = fitText(toFixed3(e.timestamp_ms), 11);
        std::string type = fitText(layerType(e.submodule_name), 14);
        std::string dev = fitText(computeDevice(e.anomaly_flag), 27);

        std::cout << "| " << id << " | "
                  << ts << " | "
                  << type << " | "
                  << dev << " |\n";
    }

    if (events.empty()) {
        std::cout << "| No traced events yet.                                             |\n";
    }

    std::cout << "+-------------------------------------------------------------------+\n";
}

static void renderAttentionPanel(int focus) {
    std::cout << "+------------------- 3. ATTENTION MATRIX VISUALIZER "
              << (focus == 2 ? "[FOCUS ACTIVE]" : "")
              << " -------------------+\n";

    std::cout << "| Tokens: [I] [want] [it] [to] [be] [keyboard] [driven]             |\n";
    std::cout << "| Viewport Window: [0-7] x [0-7]                                    |\n";
    std::cout << "|                                                                   |\n";
    std::cout << "| [I]        ##   ..   ..   ..   ..   ..   ..                       |\n";
    std::cout << "| [want]     ..   ##   ..   ..   ..   ..   ..                       |\n";
    std::cout << "| [it]       ..   ..   ##   ..   ..   ..   ..                       |\n";
    std::cout << "| [to]       ..   ..   ..   ##   ..   ..   ..                       |\n";
    std::cout << "| [be]       ..   ..   ..   ..   ##   ..   ..                       |\n";
    std::cout << "| [keyboard] ..   ..   ..   ..   ..   ##   ..                       |\n";
    std::cout << "| [driven]   ..   ..   ..   ..   ..   ..   ##                       |\n";
    std::cout << "|                                                                   |\n";
    std::cout << "| [Focus + F]: Open Fullscreen                                       |\n";
    std::cout << "| [Arrows/(h,j,k,l)]: Pan Matrix                                    |\n";
    std::cout << "| [+/-]: Change Weight Contrast                                     |\n";
    std::cout << "+-------------------------------------------------------------------+\n";
}

static void renderMetricsPanel(const Metrics* m, int focus) {
    std::cout << "+----------------------- 4. RUNTIME METRICS INSPECTOR "
              << (focus == 3 ? "[FOCUS ACTIVE]" : "")
              << " -----------------------+\n";

    if (!m) {
        std::cout << "| No active selection.                                               |\n";
        std::cout << "+-------------------------------------------------------------------+\n";
        return;
    }

    int filled = static_cast<int>(m->sparsity_rate * 20.0);
    filled = std::clamp(filled, 0, 20);

    std::cout << "| Tensor Shape : " << fitText(m->tensor_shape, 20)
              << "   Dtype: " << fitText(m->dtype, 8) << "             |\n";
    std::cout << "| Layer        : " << fitText(m->layer_name, 20)
              << "   Submodule: " << fitText(m->submodule_name, 10) << "        |\n";

    std::cout << "| Sparsity Rate: [";
    for (int i = 0; i < 20; ++i) std::cout << (i < filled ? '#' : '.');
    std::cout << "] " << std::fixed << std::setprecision(1) << (m->sparsity_rate * 100.0) << "%";

    if (m->latency_ms <= 1.6) {
        std::cout << "   Latency Delta: " << std::fixed << std::setprecision(3) << m->latency_ms
                  << " ms (Within Normal Bounds)";
    } else {
        std::cout << "   Latency Delta: " << std::fixed << std::setprecision(3) << m->latency_ms
                  << " ms (High)";
    }
    std::cout << "          |\n";

    std::cout << "| Mean Activation: " << std::fixed << std::setprecision(3) << m->mean_activation
              << "   Max Activation: " << std::fixed << std::setprecision(3) << m->max_activation
              << "                                |\n";

    std::cout << "+-------------------------------------------------------------------+\n";
}

static void renderAnomalyLedger(const std::vector<Metrics>& events, int focus) {
    std::cout << "+----------------------- 5. NUMERICAL ANOMALY LEDGER "
              << (focus == 4 ? "[FOCUS ACTIVE]" : "")
              << " -----------------------+\n";

    bool any = false;
    for (const auto& e : events) {
        if (!e.anomaly_flag) continue;
        any = true;

        std::cout << "| " << fitText(toFixed3(e.timestamp_ms), 11)
                  << "  ALERT  "
                  << fitText(e.layer_name, 10)
                  << " / "
                  << fitText(e.submodule_name, 6)
                  << "  -> Max/Latency anomaly detected"
                  << std::string(10, ' ') << "|\n";
    }

    if (!any) {
        std::cout << "| No anomalies detected.                                             |\n";
    }

    std::cout << "+-------------------------------------------------------------------+\n";
}

void runDashboard(const RingBuffer& rb) {
    int focus = 0;
    int selectedLayer = 1;

    while (true) {
        clearScreen();

        const auto events = rb.getAll();
        const Metrics* selectedMetric = pickSelectedMetric(events, selectedLayer);

        renderHelpBar(focus);
        renderTopology(focus, selectedLayer);
        renderStream(events, focus);
        renderAttentionPanel(focus);
        renderMetricsPanel(selectedMetric, focus);
        renderAnomalyLedger(events, focus);

        std::cout << "\nSelected layer: layers." << selectedLayer
                  << "   |   Focus index: " << focus
                  << "   |   Press j/k to switch layer\n";

#ifdef _WIN32
        int ch = _getch();

        if (ch == 9) {
            focus = (focus + 1) % 5;
        } else if (ch == 'q' || ch == 'Q') {
            break;
        } else if (ch == 'j' || ch == 'J') {
            selectedLayer = (selectedLayer + 1) % 3;
        } else if (ch == 'k' || ch == 'K') {
            selectedLayer = (selectedLayer + 2) % 3;
        }
#else
        char ch;
        std::cin >> ch;
        if (ch == 'q' || ch == 'Q') break;
        if (ch == '\t') focus = (focus + 1) % 5;
        if (ch == 'j' || ch == 'J') selectedLayer = (selectedLayer + 1) % 3;
        if (ch == 'k' || ch == 'K') selectedLayer = (selectedLayer + 2) % 3;
#endif
    }
}