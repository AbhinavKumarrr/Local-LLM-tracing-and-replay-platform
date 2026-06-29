#pragma once
#include <string>
#include "transformer.hpp"

namespace llmscope {

struct AppConfig {
    std::string prompt = "hello world";
    TransformerConfig model;
    int ring_capacity = 256;
    int step_delay_ms = 450;
};

int run_app(const AppConfig& cfg);

}  // namespace llmscope