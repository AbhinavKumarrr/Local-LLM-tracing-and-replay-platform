#include "../include/ring_buffer.hpp"
#include "../include/tracer.hpp"
#include "../include/dashboard.hpp"

int main() {
    RingBuffer rb(64);
    Tracer tracer(rb);

    tracer.traceLayer(101, "layers.0", "attn", "[1, 32, 4096]", "float16", 21.114, 1.142, 0.542, 0.12, 1.45);
    tracer.traceLayer(102, "layers.0", "mlp",  "[1, 32, 4096]", "float16", 21.118, 1.380, 0.538, 0.14, 1.62);
    tracer.traceLayer(103, "layers.1", "attn", "[1, 32, 4096]", "float16", 21.122, 1.510, 0.551, 0.11, 1.71);
    tracer.traceLayer(104, "layers.1", "mlp",  "[1, 32, 4096]", "float16", 21.128, 1.230, 0.547, 0.10, 1.54);
    tracer.traceLayer(105, "layers.2", "attn", "[1, 32, 4096]", "float16", 21.133, 1.660, 0.560, 0.09, 1.80);

    runDashboard(rb);
    return 0;
}