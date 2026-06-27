#include "../include/ring_buffer.hpp"

RingBuffer::RingBuffer(std::size_t cap)
    : buffer(cap), capacity(cap), start_index(0), count(0) {}

void RingBuffer::push(const Metrics& m) {
    if (capacity == 0) return;

    if (count < capacity) {
        buffer[(start_index + count) % capacity] = m;
        ++count;
    } else {
        buffer[start_index] = m;
        start_index = (start_index + 1) % capacity;
    }
}

std::vector<Metrics> RingBuffer::getAll() const {
    std::vector<Metrics> result;
    result.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(buffer[(start_index + i) % capacity]);
    }

    return result;
}

std::size_t RingBuffer::size() const {
    return count;
}

bool RingBuffer::empty() const {
    return count == 0;
}

std::size_t RingBuffer::getCapacity() const {
    return capacity;
}