#include "../include/ring_buffer.hpp"
#include "../include/tracer.hpp"
#include "../include/dashboard.hpp"
#include "../include/replay.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

static void seedDemoTrace(Tracer& tracer) {
    tracer.traceLayer(101, "layers.0", "attn", "[1, 32, 4096]", "float16", 21.114, 1.142, 0.542, 0.12, 1.45);
    tracer.traceLayer(102, "layers.0", "mlp",  "[1, 32, 4096]", "float16", 21.118, 1.380, 0.538, 0.14, 1.62);
    tracer.traceLayer(103, "layers.1", "attn", "[1, 32, 4096]", "float16", 21.122, 1.510, 0.551, 0.11, 1.71);
    tracer.traceLayer(104, "layers.1", "mlp",  "[1, 32, 4096]", "float16", 21.128, 1.230, 0.547, 0.10, 1.54);
    tracer.traceLayer(105, "layers.2", "attn", "[1, 32, 4096]", "float16", 21.133, 1.660, 0.560, 0.09, 1.80);
}

int main(int argc, char** argv) {
    if (argc >= 3 && string(argv[1]) == "--replay") {
        string path = argv[2];
        vector<Metrics> events = ReplayEngine::loadSession(path);

        if (events.empty()) {
            cerr << "No events loaded from: " << path << '\n';
            return 1;
        }

        RingBuffer rb(64);
        ReplayEngine::replayToBuffer(events, rb, 0);
        runDashboard(rb);
        return 0;
    }

    RingBuffer rb(64);
    Tracer tracer(rb);

    seedDemoTrace(tracer);

    vector<Metrics> events = rb.getAll();

    string record_path = "traceformer_session.trace";
    if (argc >= 3 && string(argv[1]) == "--record") {
        record_path = argv[2];
    }

    if (!ReplayEngine::saveSession(record_path, events)) {
        cerr << "Failed to save session to: " << record_path << '\n';
    } else {
        cout << "Session saved to: " << record_path << '\n';
    }

    runDashboard(rb);
    return 0;
}