#include "../include/tracer.hpp"

Tracer::Tracer(RingBuffer& rb)
    : buffer(rb), event_counter(0) {}

void Tracer::traceLayer(
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
) {
    ++event_counter;

    constexpr double LATENCY_THRESHOLD = 1.6;
    constexpr double MAX_ACTIVATION_THRESHOLD = 1.75;

    bool anomaly = (latency_ms > LATENCY_THRESHOLD || max_activation > MAX_ACTIVATION_THRESHOLD);

    Metrics m{
        static_cast<int>(event_counter),
        token_id,
        layer_name,
        submodule_name,
        tensor_shape,
        dtype,
        timestamp_ms,
        latency_ms,
        sparsity_rate,
        mean_activation,
        max_activation,
        anomaly
    };

    buffer.push(m);
}