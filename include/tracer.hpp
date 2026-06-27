#pragma once
#include <string>
#include <cstddef>
#include "ring_buffer.hpp"

class Tracer {
private:
    RingBuffer& buffer;
    std::size_t event_counter;

public:
    explicit Tracer(RingBuffer& rb);

    void traceLayer(
        int token_id,
        const std::string& layer_name,
        const std::string& submodule_name,
        const std::string& tensor_shape,
        const std::string& dtype,
        double timestamp_ms,
        double latency_ms,
        double sparsity_rate,
        double mean_activation,
        double max_activation
    );
};