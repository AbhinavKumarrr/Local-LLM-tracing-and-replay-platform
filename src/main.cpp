#include "../include/tui.hpp"
#include "../include/ring_buffer.hpp"

int main() {
    RingBuffer rb(5);

    rb.push({"embed_tokens", 101, 1.2, 52.0, "[1,32,4096]"});
    rb.push({"layers.0", 102, 1.8, 55.0, "[1,32,4096]"});
    rb.push({"layers.1", 103, 1.4, 53.0, "[1,32,4096]"});

    renderUI(rb);

    return 0;
}