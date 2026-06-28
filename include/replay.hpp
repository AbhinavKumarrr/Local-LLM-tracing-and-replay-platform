#pragma once

#include <string>
#include <vector>
#include "metrics.hpp"
#include "ring_buffer.hpp"

class ReplayEngine {
public:
    static bool saveSession(const std::string& path, const std::vector<Metrics>& events);
    static std::vector<Metrics> loadSession(const std::string& path);
    static void replayToBuffer(const std::vector<Metrics>& events, RingBuffer& rb, int delay_ms = 0);
};