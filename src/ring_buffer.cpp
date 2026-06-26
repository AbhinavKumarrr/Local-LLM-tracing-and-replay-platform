#include "../include/ring_buffer.hpp"

RingBuffer::RingBuffer(int cap) {
    capacity = cap;
    index = 0;
}

void RingBuffer::push(Metrics m) {
    if (buffer.size() < capacity) {
        buffer.push_back(m);
    } else {
        buffer[index] = m;
        index = (index + 1) % capacity;
    }
}

vector<Metrics> RingBuffer::getAll() {
    return buffer;
}