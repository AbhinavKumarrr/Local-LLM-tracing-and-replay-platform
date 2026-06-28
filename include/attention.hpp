#pragma once
#include <string>
#include <vector>
#include "metrics.hpp"

struct AttentionState {
    std::vector<std::string> tokens;
    std::vector<std::vector<double>> weights;
    int row_offset = 0;
    int col_offset = 0;
    int window = 7;
    double contrast = 1.0;
    bool fullscreen = false;
};

AttentionState buildAttentionState(const Metrics& metric);
void panAttention(AttentionState& state, int d_row, int d_col);
void adjustContrast(AttentionState& state, double delta);
void toggleFullscreen(AttentionState& state);
void clampAttentionView(AttentionState& state);