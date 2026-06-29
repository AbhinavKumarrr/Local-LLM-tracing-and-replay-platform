#include "llmscope/app.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

namespace {
bool next_arg(int argc, char** argv, int& i, string& out) {
    if (i + 1 >= argc) return false;
    out = argv[++i];
    return true;
}

void print_usage(const char* argv0) {
    cout
        << "llm_scope - Local LLM Instrumentation, Tracing and Replay Platform\n\n"
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --prompt <text>     Prompt to trace through the model\n"
        << "  --layers <n>        Number of transformer layers (default 8)\n"
        << "  --heads <n>         Attention heads (default 8)\n"
        << "  --dmodel <n>        Model hidden size (default 128)\n"
        << "  --dff <n>           Feed-forward size (default 512)\n"
        << "  --buffer <n>        Ring-buffer capacity for events (default 256)\n"
        << "  --delay <ms>        Live capture step delay in ms (default 450)\n"
        << "  --name <text>       Model display name\n"
        << "  -h, --help          Show this help\n";
}
}  // namespace

int main(int argc, char** argv) {
    llmscope::AppConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const string a = argv[i];
        string v;

        auto need = [&](const char* flag) -> bool {
            if (!next_arg(argc, argv, i, v)) {
                cerr << "error: missing value for " << flag << '\n';
                return false;
            }
            return true;
        };

        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--prompt") {
            if (!need("--prompt")) return 2;
            cfg.prompt = v;
        } else if (a == "--layers") {
            if (!need("--layers")) return 2;
            cfg.model.n_layers = max(1, atoi(v.c_str()));
        } else if (a == "--heads") {
            if (!need("--heads")) return 2;
            cfg.model.n_heads = max(1, atoi(v.c_str()));
        } else if (a == "--dmodel") {
            if (!need("--dmodel")) return 2;
            cfg.model.d_model = max(8, atoi(v.c_str()));
        } else if (a == "--dff") {
            if (!need("--dff")) return 2;
            cfg.model.d_ff = max(16, atoi(v.c_str()));
        } else if (a == "--buffer") {
            if (!need("--buffer")) return 2;
            cfg.ring_capacity = max(8, atoi(v.c_str()));
        } else if (a == "--delay") {
            if (!need("--delay")) return 2;
            cfg.step_delay_ms = max(0, atoi(v.c_str()));
        } else if (a == "--name") {
            if (!need("--name")) return 2;
            cfg.model.model_name = v;
        }
    }

    return llmscope::run_app(cfg);
}