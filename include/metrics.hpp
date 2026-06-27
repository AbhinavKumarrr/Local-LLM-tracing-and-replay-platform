#pragma once
#include <string>

struct Metrics {
    int event_id;
    int token_id;
    std::string layer_name;
    std::string submodule_name;
    std::string tensor_shape;
    std::string dtype;
    double timestamp_ms;
    double latency_ms;
    double sparsity_rate;
    double mean_activation;
    double max_activation;
    bool anomaly_flag;
};