#pragma once
#include <vector>
#include "metrics.hpp"

class RingBuffer {
private:
    vector<Metrics> buffer;
    int capacity;
    int index;

public:
    RingBuffer(int cap);

    void push(Metrics m);

    vector<Metrics> getAll();
};