#include "../include/dashboard.hpp"
#include "../include/attention.hpp"

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

using namespace std;

static void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

static string fitText(const string& s, size_t width) {
    if (s.size() >= width) return s.substr(0, width);
    return s + string(width - s.size(), ' ');
}

static string toFixed3(double v) {
    ostringstream oss;
    oss << fixed << setprecision(3) << v;
    return oss.str();
}

static string layerType(const string& submodule) {
    if (submodule == "attn") return "Attn (Self)";
    if (submodule == "mlp") return "MLP (SwiGLU)";
    return submodule;
}

static string computeDevice(bool anomaly) {
    return anomaly ? "CPU (Fallback)" : "CUDA [GPU 0]";
}

static int extractLayerIndex(const string& layer_name) {
    string digits;
    bool found = false;

    for (char c : layer_name) {
        if (c >= '0' && c <= '9') {
            digits += c;
            found = true;
        } else if (found) {
            break;
        }
    }

    if (digits.empty()) return 0;
    return stoi(digits);
}

static const Metrics* pickSelectedMetric(const vector<Metrics>& events, int selectedLayer) {
    const string target = "layers." + to_string(selectedLayer);

    for (const auto& e : events) {
        if (e.layer_name.find(target) != string::npos) {
            return &e;
        }
    }

    if (!events.empty()) return &events.back();
    return nullptr;
}

static string cellForWeight(double w, double contrast) {
    double v = w * contrast;
    if (v > 1.0) v = 1.0;
    if (v < 0.0) v = 0.0;

    if (v >= 0.85) return "##";
    if (v >= 0.65) return "++";
    if (v >= 0.45) return "--";
    if (v >= 0.25) return "..";
    return "  ";
}

static void renderHelpBar(int focus, const AttentionState& attn) {
    static const char* names[] = {
        "1. MODEL TOPOLOGY",
        "2. LIVE PACKET STREAM",
        "3. ATTENTION MATRIX",
        "4. RUNTIME METRICS",
        "5. ANOMALY LEDGER"
    };

    cout << "[Tab]: Cycle Focus  |  [j/k]: Move Cursor or Pan  |  [Space]: Select Layer  |  [h/l]: Pan  |  [+/-]: Contrast  |  [F]: Fullscreen  |  [Q]: Quit App\n";
    cout << "Focus: " << names[focus] << "   "
         << "[Matrix " << (attn.fullscreen ? "Fullscreen" : "Windowed") << "]\n\n";
}

static void renderTopology(int focus, int cursorLayer, int selectedLayer) {
    cout << "+------------------------- 1. MODEL TOPOLOGY "
         << (focus == 0 ? "[FOCUS ACTIVE]" : "")
         << " -------------------------+\n";

    cout << "| > llama-3-8b                                                     |\n";
    cout << "|   > embed_tokens                                                  |\n";
    cout << "|   > layers                                                        |\n";

    for (int i = 0; i < 3; ++i) {
        bool cursor = (i == cursorLayer);
        bool active = (i == selectedLayer);

        cout << "|   " << (cursor ? ">" : " ") << " ";
        cout << "layers." << i;

        if (active) {
            cout << "  [Active Capture Target]";
        } else if (cursor) {
            cout << "  [Cursor]";
        }

        cout << string(45, ' ') << "|\n";
    }

    cout << "+-------------------------------------------------------------------+\n";
}

static void renderStream(const vector<Metrics>& events, int focus) {
    cout << "+---------------------- 2. LIVE PACKET STREAM "
         << (focus == 1 ? "[FOCUS ACTIVE]" : "")
         << " ----------------------+\n";
    cout << "| ID   | TIMESTAMP   | LAYER TYPE     | COMPUTE DEVICE              |\n";
    cout << "+------+-------------+----------------+-----------------------------+\n";

    for (const auto& e : events) {
        string id = fitText(to_string(e.event_id), 4);
        string ts = fitText(toFixed3(e.timestamp_ms), 11);
        string type = fitText(layerType(e.submodule_name), 14);
        string dev = fitText(computeDevice(e.anomaly_flag), 27);

        cout << "| " << id << " | "
             << ts << " | "
             << type << " | "
             << dev << " |\n";
    }

    if (events.empty()) {
        cout << "| No traced events yet.                                             |\n";
    }

    cout << "+-------------------------------------------------------------------+\n";
}

static void renderAttentionPanel(const AttentionState& state, int focus) {
    cout << "+------------------- 3. ATTENTION MATRIX VISUALIZER "
         << (focus == 2 ? "[FOCUS ACTIVE]" : "")
         << " -------------------+\n";

    const int n = static_cast<int>(state.tokens.size());
    const int row_start = state.row_offset;
    const int col_start = state.col_offset;
    const int row_end = min(n, row_start + state.window);
    const int col_end = min(n, col_start + state.window);

    cout << "| Tokens: ";
    for (int j = col_start; j < col_end; ++j) {
        cout << "[" << fitText(state.tokens[j], 8) << "] ";
    }
    cout << string(32, ' ') << "|\n";

    cout << "| Viewport Window: [" << row_start << "-" << max(row_start, row_end - 1)
         << "] x [" << col_start << "-" << max(col_start, col_end - 1) << "]";
    if (state.fullscreen) cout << "   (Fullscreen)";
    cout << string(20, ' ') << "|\n";

    for (int i = row_start; i < row_end; ++i) {
        cout << "| ";
        cout << "[" << fitText(state.tokens[i], 8) << "] ";

        for (int j = col_start; j < col_end; ++j) {
            cout << cellForWeight(state.weights[i][j], state.contrast) << "  ";
        }

        cout << string(16, ' ') << "|\n";
    }

    cout << "|                                                                   |\n";
    cout << "| [Focus + F]: Open Fullscreen                                       |\n";
    cout << "| [Arrows/(h,j,k,l)]: Pan Matrix                                    |\n";
    cout << "| [+/-]: Change Weight Contrast                                     |\n";
    cout << "+-------------------------------------------------------------------+\n";
}

static void renderMetricsPanel(const Metrics* m, int focus) {
    cout << "+----------------------- 4. RUNTIME METRICS INSPECTOR "
         << (focus == 3 ? "[FOCUS ACTIVE]" : "")
         << " -----------------------+\n";

    if (!m) {
        cout << "| No active selection.                                             |\n";
        cout << "+-------------------------------------------------------------------+\n";
        return;
    }

    int filled = static_cast<int>(m->sparsity_rate * 20.0);
    filled = max(0, min(20, filled));

    cout << "| Tensor Shape : " << fitText(m->tensor_shape, 20)
         << "   Dtype: " << fitText(m->dtype, 8) << "             |\n";
    cout << "| Layer        : " << fitText(m->layer_name, 20)
         << "   Submodule: " << fitText(m->submodule_name, 10) << "        |\n";

    cout << "| Sparsity Rate: [";
    for (int i = 0; i < 20; ++i) cout << (i < filled ? '#' : '.');
    cout << "] " << fixed << setprecision(1) << (m->sparsity_rate * 100.0) << "%";

    if (m->latency_ms <= 1.6) {
        cout << "   Latency Delta: " << fixed << setprecision(3) << m->latency_ms
             << " ms (Within Normal Bounds)";
    } else {
        cout << "   Latency Delta: " << fixed << setprecision(3) << m->latency_ms
             << " ms (High)";
    }
    cout << "          |\n";

    cout << "| Mean Activation: " << fixed << setprecision(3) << m->mean_activation
         << "   Max Activation: " << fixed << setprecision(3) << m->max_activation
         << "                                |\n";

    cout << "+-------------------------------------------------------------------+\n";
}

static void renderAnomalyLedger(const vector<Metrics>& events, int focus) {
    cout << "+----------------------- 5. NUMERICAL ANOMALY LEDGER "
         << (focus == 4 ? "[FOCUS ACTIVE]" : "")
         << " -----------------------+\n";

    bool any = false;
    for (const auto& e : events) {
        if (!e.anomaly_flag) continue;
        any = true;

        cout << "| " << fitText(toFixed3(e.timestamp_ms), 11)
             << "  ALERT  "
             << fitText(e.layer_name, 10)
             << " / "
             << fitText(e.submodule_name, 6)
             << "  -> Max/Latency anomaly detected"
             << string(10, ' ') << "|\n";
    }

    if (!any) {
        cout << "| No anomalies detected.                                           |\n";
    }

    cout << "+-------------------------------------------------------------------+\n";
}

void runDashboard(const RingBuffer& rb) {
    int focus = 0;
    int cursorLayer = 1;
    int selectedLayer = 1;
    AttentionState attention;
    int attentionLayer = -1;

    while (true) {
        const auto events = rb.getAll();
        const Metrics* selectedMetric = pickSelectedMetric(events, selectedLayer);

        if (selectedMetric) {
            int metricLayer = extractLayerIndex(selectedMetric->layer_name);

            if (attentionLayer != metricLayer || attention.tokens.empty()) {
                AttentionState fresh = buildAttentionState(*selectedMetric);
                fresh.row_offset = attention.row_offset;
                fresh.col_offset = attention.col_offset;
                fresh.contrast = attention.contrast;
                fresh.fullscreen = attention.fullscreen;
                clampAttentionView(fresh);
                attention = fresh;
                attentionLayer = metricLayer;
            }
        }

        clearScreen();
        renderHelpBar(focus, attention);
        renderTopology(focus, cursorLayer, selectedLayer);
        renderStream(events, focus);
        renderAttentionPanel(attention, focus);
        renderMetricsPanel(selectedMetric, focus);
        renderAnomalyLedger(events, focus);

        cout << "\nCursor layer: layers." << cursorLayer
             << "   |   Selected layer: layers." << selectedLayer
             << "   |   Focus index: " << focus << "\n";

#ifdef _WIN32
        int ch = _getch();

        if (ch == 0 || ch == 224) {
            int arrow = _getch();
            if (focus == 2) {
                if (arrow == 72) panAttention(attention, -1, 0);
                if (arrow == 80) panAttention(attention, 1, 0);
                if (arrow == 75) panAttention(attention, 0, -1);
                if (arrow == 77) panAttention(attention, 0, 1);
                clampAttentionView(attention);
            }
        } else if (ch == 9) {
            focus = (focus + 1) % 5;
        } else if (ch == 'q' || ch == 'Q') {
            break;
        } else if (ch == 'f' || ch == 'F') {
            toggleFullscreen(attention);
        } else if (ch == '+' || ch == '=') {
            adjustContrast(attention, 0.10);
        } else if (ch == '-' || ch == '_') {
            adjustContrast(attention, -0.10);
        } else if (ch == ' ') {
            if (focus == 0) {
                selectedLayer = cursorLayer;
            }
        } else if (focus == 2) {
            if (ch == 'h' || ch == 'H') panAttention(attention, 0, -1);
            else if (ch == 'l' || ch == 'L') panAttention(attention, 0, 1);
            else if (ch == 'j' || ch == 'J') panAttention(attention, 1, 0);
            else if (ch == 'k' || ch == 'K') panAttention(attention, -1, 0);
            clampAttentionView(attention);
        } else if (focus == 0) {
            if (ch == 'j' || ch == 'J') cursorLayer = (cursorLayer + 1) % 3;
            else if (ch == 'k' || ch == 'K') cursorLayer = (cursorLayer + 2) % 3;
        }
#else
        char ch;
        cin >> ch;

        if (ch == 'q' || ch == 'Q') break;
        if (ch == '\t') focus = (focus + 1) % 5;
        if (ch == 'f' || ch == 'F') toggleFullscreen(attention);
        if (ch == '+' || ch == '=') adjustContrast(attention, 0.10);
        if (ch == '-' || ch == '_') adjustContrast(attention, -0.10);

        if (ch == ' ' && focus == 0) {
            selectedLayer = cursorLayer;
        } else if (focus == 2) {
            if (ch == 'h' || ch == 'H') panAttention(attention, 0, -1);
            else if (ch == 'l' || ch == 'L') panAttention(attention, 0, 1);
            else if (ch == 'j' || ch == 'J') panAttention(attention, 1, 0);
            else if (ch == 'k' || ch == 'K') panAttention(attention, -1, 0);
            clampAttentionView(attention);
        } else if (focus == 0) {
            if (ch == 'j' || ch == 'J') cursorLayer = (cursorLayer + 1) % 3;
            else if (ch == 'k' || ch == 'K') cursorLayer = (cursorLayer + 2) % 3;
        }
#endif
    }
}