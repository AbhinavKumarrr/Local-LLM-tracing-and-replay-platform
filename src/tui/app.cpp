#include "llmscope/app.hpp"

#include "llmscope/event.hpp"
#include "llmscope/instrumentation.hpp"
#include "llmscope/model_graph.hpp"
#include "llmscope/ring_buffer.hpp"
#include "llmscope/tokenizer.hpp"
#include "llmscope/transformer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#endif

using namespace std;

namespace llmscope {

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

static string fmt3(double v) {
    ostringstream oss;
    oss << fixed << setprecision(3) << v;
    return oss.str();
}

static string kindName(ModuleKind k) {
    if (k == ModuleKind::Embedding) return "embedding";
    if (k == ModuleKind::LayerNorm) return "layernorm";
    if (k == ModuleKind::Attention) return "attention";
    if (k == ModuleKind::AttentionScores) return "attn_scores";
    if (k == ModuleKind::MLP) return "mlp";
    if (k == ModuleKind::Residual) return "residual";
    if (k == ModuleKind::LogitsHead) return "logits";
    return "other";
}

class Dashboard : public TraceSink {
public:
    Dashboard(const AppConfig& cfg, ModelGraph graph)
        : cfg_(cfg), graph_(std::move(graph)), stream_(static_cast<size_t>(cfg.ring_capacity)) {
        nodes_ = graph_.flatten();
        cursor_ = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i]->kind == ModuleKind::Attention) {
                cursor_ = static_cast<int>(i);
                break;
            }
        }
    }

    void on_event(const TraceEvent& ev) override {
        lock_guard<mutex> lk(mutex_);
        stream_.push(ev);
        latest_by_path_[ev.module_path] = ev;
        if (ev.kind == ModuleKind::Attention && ev.layer_index >= 0) {
            latest_attn_by_layer_[ev.layer_index] = ev;
        }
        if (ev.stats.has_nan || ev.stats.has_inf || ev.stats.abs_max > 6.0f || ev.stats.sparsity > 0.92) {
            anomalies_.push_back(ev);
            if (anomalies_.size() > 12) {
                anomalies_.erase(anomalies_.begin());
            }
        }
        total_events_++;
    }

    void handle_key(int ch) {
        if (ch == 'q' || ch == 'Q') {
            should_quit_ = true;
            return;
        }
        if (ch == 9) {
            focus_ = (focus_ + 1) % 5;
            return;
        }
        if (ch == 'f' || ch == 'F') {
            fullscreen_ = !fullscreen_;
            return;
        }
        if (ch == '+' || ch == '=') {
            if (attention_contrast_ < 3.0) attention_contrast_ += 0.1;
            return;
        }
        if (ch == '-' || ch == '_') {
            if (attention_contrast_ > 0.3) attention_contrast_ -= 0.1;
            return;
        }

        if (focus_ == 0) {
            if (!nodes_.empty()) {
                if (ch == 'j' || ch == 'J') cursor_ = (cursor_ + 1) % static_cast<int>(nodes_.size());
                else if (ch == 'k' || ch == 'K') cursor_ = (cursor_ - 1 + static_cast<int>(nodes_.size())) % static_cast<int>(nodes_.size());
                else if (ch == ' ') selected_layer_ = activeLayer();
            }
        } else if (focus_ == 2) {
            if (ch == 'h' || ch == 'H') panX_ = max(0, panX_ - 1);
            else if (ch == 'l' || ch == 'L') panX_ += 1;
            else if (ch == 'j' || ch == 'J') panY_ += 1;
            else if (ch == 'k' || ch == 'K') panY_ = max(0, panY_ - 1);
        }
    }

    bool should_quit() const { return should_quit_; }

    string render() const {
        lock_guard<mutex> lk(mutex_);
        ostringstream out;

        const TraceEvent* active = activeEventLocked();
        out << "[Tab]: Cycle Focus  |  [j/k]: Move Cursor or Pan  |  [Space]: Select Layer  |  [+/-]: Contrast  |  [F]: Fullscreen  |  [Q]: Quit App\n";
        out << "Focus: " << focusName() << "   [Matrix " << (fullscreen_ ? "Fullscreen" : "Windowed") << "]\n\n";

        renderTopology(out);
        renderStream(out);
        renderAttention(out, active);
        renderMetrics(out, active);
        renderAnomalies(out);

        out << "\nCursor layer: layers." << activeLayer()
            << "   |   Selected layer: layers." << selected_layer_
            << "   |   Focus index: " << focus_ << '\n';

        return out.str();
    }

    void run_engine(Transformer& model,
                    const vector<int>& tokens,
                    const vector<string>& token_strs,
                    atomic<bool>& running) {
        for (size_t step = 1; step <= tokens.size() && running.load(); ++step) {
            vector<int> prefix(tokens.begin(), tokens.begin() + static_cast<long>(step));
            vector<string> prefix_strs(token_strs.begin(), token_strs.begin() + static_cast<long>(step));
            model.forward(prefix, prefix_strs, static_cast<int>(step));
            this_thread::sleep_for(chrono::milliseconds(cfg_.step_delay_ms));
        }
        running.store(false);
    }

private:
    const char* focusName() const {
        switch (focus_) {
            case 0: return "1. MODEL TOPOLOGY";
            case 1: return "2. LIVE PACKET STREAM";
            case 2: return "3. ATTENTION MATRIX";
            case 3: return "4. RUNTIME METRICS";
            case 4: return "5. ANOMALY LEDGER";
            default: return "UNKNOWN";
        }
    }

    int activeLayer() const {
        if (nodes_.empty()) return 0;
        const GraphNode* n = nodes_[static_cast<size_t>(cursor_)];
        return n->layer_index >= 0 ? n->layer_index : 0;
    }

    const TraceEvent* activeEventLocked() const {
        const int layer = activeLayer();
        auto it = latest_by_path_.find("layers." + std::to_string(layer) + ".attn");
        if (it != latest_by_path_.end()) return &it->second;
        it = latest_by_path_.find("layers." + std::to_string(layer) + ".mlp");
        if (it != latest_by_path_.end()) return &it->second;
        if (!latest_by_path_.empty()) return &latest_by_path_.begin()->second;
        return nullptr;
    }

    void renderTopology(ostringstream& out) const {
        out << "+------------------------- 1. MODEL TOPOLOGY "
            << (focus_ == 0 ? "[FOCUS ACTIVE]" : "")
            << " -------------------------+\n";
        out << "| > " << fitText(graph_.root.name, 58) << " |\n";
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const GraphNode* n = nodes_[i];
            if (n->depth > 3) continue;
            string indent(static_cast<size_t>(n->depth) * 2, ' ');
            bool cur = static_cast<int>(i) == cursor_;
            bool sel = n->layer_index == selected_layer_ && n->layer_index >= 0;
            out << "| " << (cur ? ">" : " ") << " " << fitText(indent + n->path, 58);
            if (sel) out << " [Active Capture Target]";
            out << " |\n";
            if (i > 10) break;
        }
        out << "+-------------------------------------------------------------------+\n";
    }

    void renderStream(ostringstream& out) const {
        auto events = stream_.snapshot();
        out << "+---------------------- 2. LIVE PACKET STREAM "
            << (focus_ == 1 ? "[FOCUS ACTIVE]" : "")
            << " ----------------------+\n";
        out << "| ID   | STEP | MODULE                     | SHAPE          | LAT ms |\n";
        out << "+------+------ +----------------------------+----------------+--------+\n";

        const int start = max(0, static_cast<int>(events.size()) - 8);
        for (int i = start; i < static_cast<int>(events.size()); ++i) {
            const auto& e = events[static_cast<size_t>(i)];
            out << "| " << fitText(std::to_string(e.id), 4) << " | "
                << fitText(std::to_string(e.token_step), 4) << " | "
                << fitText(e.module_path, 26) << " | "
                << fitText(e.stats.shape_string(), 14) << " | "
                << fitText(fmt3(e.latency_ms), 6) << " |\n";
        }
        if (events.empty()) {
            out << "| No traced events yet.                                             |\n";
        }
        out << "+-------------------------------------------------------------------+\n";
    }

    void renderAttention(ostringstream& out, const TraceEvent* active) const {
        out << "+------------------- 3. ATTENTION MATRIX VISUALIZER "
            << (focus_ == 2 ? "[FOCUS ACTIVE]" : "")
            << " -------------------+\n";

        if (!active || !active->has_attention()) {
            out << "| waiting for attention capture.                                    |\n";
            out << "| navigate to a layer's attn node (j/k)                            |\n";
            out << "+-------------------------------------------------------------------+\n";
            return;
        }

        const int heads = max(1, active->num_heads);
        const int head = min(attn_head_, heads - 1);
        const int seq = active->seq_len;
        const auto& matrix = active->attention[static_cast<size_t>(head)];

        out << "| module: " << fitText(active->module_path, 24)
            << " head " << head << "/" << (heads - 1)
            << " seq=" << seq
            << (fullscreen_ ? "  [Fullscreen]" : "")
            << " |\n";

        const int max_show = min(seq, 7);
        out << "| Tokens: ";
        for (int k = 0; k < max_show; ++k) {
            if (k < static_cast<int>(active->tokens.size()))
                out << "[" << fitText(active->tokens[static_cast<size_t>(k)], 5) << "] ";
            else
                out << "[tok] ";
        }
        out << "|\n";

        for (int q = 0; q < max_show; ++q) {
            out << "| [" << fitText(active->tokens[static_cast<size_t>(q)], 5) << "] ";
            for (int k = 0; k < max_show; ++k) {
                float p = 0.0f;
                if (q < seq && k < seq) p = matrix[static_cast<size_t>(q) * seq + k];
                double v = min(1.0, max(0.0, static_cast<double>(p) * attention_contrast_));
                if (v >= 0.85) out << "## ";
                else if (v >= 0.65) out << "++ ";
                else if (v >= 0.45) out << "-- ";
                else if (v >= 0.25) out << ".. ";
                else out << "   ";
            }
            out << "|\n";
        }

        out << "| [Focus + F]: Open Fullscreen                                       |\n";
        out << "| [Arrows/(h,j,k,l)]: Pan Matrix                                    |\n";
        out << "| [+/-]: Change Weight Contrast                                     |\n";
        out << "+-------------------------------------------------------------------+\n";
    }

    void renderMetrics(ostringstream& out, const TraceEvent* active) const {
        out << "+----------------------- 4. RUNTIME METRICS INSPECTOR "
            << (focus_ == 3 ? "[FOCUS ACTIVE]" : "")
            << " -----------------------+\n";

        if (!active) {
            out << "| No active selection.                                             |\n";
            out << "+-------------------------------------------------------------------+\n";
            return;
        }

        int filled = static_cast<int>(active->stats.sparsity * 20.0);
        filled = max(0, min(20, filled));

        out << "| Tensor Shape : " << fitText(active->stats.shape_string(), 20)
            << "   Dtype: " << fitText(active->stats.dtype, 8) << "             |\n";
        out << "| Layer        : " << fitText(active->module_path, 20)
            << "   Kind: " << fitText(kindName(active->kind), 10) << "        |\n";

        out << "| Sparsity Rate: [";
        for (int i = 0; i < 20; ++i) out << (i < filled ? '#' : '.');
        out << "] " << fixed << setprecision(1) << (active->stats.sparsity * 100.0) << "%";

        if (active->latency_ms <= 1.6) {
            out << "   Latency Delta: " << fixed << setprecision(3) << active->latency_ms
                << " ms (Within Normal Bounds)";
        } else {
            out << "   Latency Delta: " << fixed << setprecision(3) << active->latency_ms
                << " ms (High)";
        }
        out << "          |\n";

        out << "| Mean Activation: " << fixed << setprecision(3) << active->stats.mean
            << "   Max Activation: " << fixed << setprecision(3) << active->stats.max
            << "                                |\n";
        out << "+-------------------------------------------------------------------+\n";
    }

    void renderAnomalies(ostringstream& out) const {
        out << "+----------------------- 5. NUMERICAL ANOMALY LEDGER "
            << (focus_ == 4 ? "[FOCUS ACTIVE]" : "")
            << " -----------------------+\n";

        if (anomalies_.empty()) {
            out << "| No anomalies detected.                                           |\n";
            out << "+-------------------------------------------------------------------+\n";
            return;
        }

        const int start = max(0, static_cast<int>(anomalies_.size()) - 6);
        for (int i = start; i < static_cast<int>(anomalies_.size()); ++i) {
            const auto& e = anomalies_[static_cast<size_t>(i)];
            out << "| " << fitText(fmt3(static_cast<double>(e.timestamp_ns) / 1'000'000.0), 11)
                << "  ALERT  "
                << fitText(e.module_path, 18)
                << "  -> max/sparsity/NaN anomaly"
                << string(10, ' ') << "|\n";
        }
        out << "+-------------------------------------------------------------------+\n";
    }

private:
    AppConfig cfg_;
    ModelGraph graph_;
    RingBuffer<TraceEvent> stream_;
    vector<const GraphNode*> nodes_;
    mutable mutex mutex_;
    unordered_map<string, TraceEvent> latest_by_path_;
    unordered_map<int, TraceEvent> latest_attn_by_layer_;
    vector<TraceEvent> anomalies_;
    int cursor_ = 0;
    int selected_layer_ = 0;
    int focus_ = 0;
    int attn_head_ = 0;
    int panX_ = 0;
    int panY_ = 0;
    double attention_contrast_ = 1.0;
    bool fullscreen_ = false;
    bool should_quit_ = false;
    size_t total_events_ = 0;
};

int run_app(const AppConfig& cfg) {
    Tokenizer tok;
    vector<int> tokens = tok.encode(cfg.prompt);
    if (tokens.empty()) tokens = tok.encode("hello world");

    vector<string> token_strs;
    token_strs.reserve(tokens.size());
    for (int id : tokens) token_strs.push_back(tok.decode(id));

    ModelGraph graph = ModelGraph::build_reference(cfg.model.model_name, cfg.model.n_layers);

    HookRegistry registry;
    Dashboard dash(cfg, std::move(graph));
    registry.attach(&dash);

    Transformer model(cfg.model, &registry);

    atomic<bool> running{true};
    thread engine([&] {
        dash.run_engine(model, tokens, token_strs, running);
    });

    while (!dash.should_quit() && running.load()) {
        clearScreen();
        cout << dash.render() << flush;

#ifdef _WIN32
        if (_kbhit()) {
            int ch = _getch();
            dash.handle_key(ch);
        }
#else
        char ch;
        if (cin.get(ch)) {
            dash.handle_key(ch);
        }
#endif
        this_thread::sleep_for(chrono::milliseconds(120));
    }

    running.store(false);
    if (engine.joinable()) engine.join();
    registry.detach(&dash);
    return 0;
}

}  // namespace llmscope