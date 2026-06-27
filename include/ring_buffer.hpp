#pragma once
#include <vector>
#include <cstddef>
#include "metrics.hpp"

class RingBuffer {
private:
    std::vector<Metrics> buffer;
    std::size_t capacity;
    std::size_t start_index;
    std::size_t count;

public:
    explicit RingBuffer(std::size_t cap);

    void push(const Metrics& m);
    std::vector<Metrics> getAll() const;
    std::size_t size() const;
    bool empty() const;
    std::size_t getCapacity() const;
};