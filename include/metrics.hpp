#pragma once
#include <string>
using namespace std;

struct Metrics {
    string layer_name;
    int token_id;
    float latency_ms;
    float sparsity;
    string tensor_shape;
};