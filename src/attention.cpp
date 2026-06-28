#include "../include/attention.hpp"

#include <algorithm>
#include <cmath>

using namespace std;

static int parseLayerIndex(const string& layer_name) {
    int value = 0;
    bool found_digit = false;

    for (char c : layer_name) {
        if (c >= '0' && c <= '9') {
            found_digit = true;
            value = value * 10 + (c - '0');
        } else if (found_digit) {
            break;
        }
    }

    return value;
}

static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

AttentionState buildAttentionState(const Metrics& metric) {
    AttentionState state;

    state.tokens = {"I", "want", "it", "to", "be", "keyboard", "driven"};
    const int n = static_cast<int>(state.tokens.size());

    state.weights.assign(n, vector<double>(n, 0.0));

    const int layer_index = parseLayerIndex(metric.layer_name);
    const double layer_bias = 1.0 + 0.06 * layer_index;
    const double submodule_bias = (metric.submodule_name == "attn") ? 1.15 : 0.95;
    const double activity_bias = 0.18 + metric.mean_activation * 0.7 + metric.sparsity_rate * 0.15;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int dist = std::abs(i - j);

            double v = std::exp(-0.95 * dist);
            v *= layer_bias;
            v *= submodule_bias;

            if (i == j) v = 1.0;
            else if (dist == 1) v += 0.18;
            else if (dist == 2) v += 0.08;

            if (j > i) v += 0.03 * activity_bias;
            if (i % 2 == 0) v += 0.02 * layer_index;

            state.weights[i][j] = clamp01(v);
        }
    }

    state.row_offset = 0;
    state.col_offset = 0;
    state.window = 7;
    state.contrast = 1.0;
    state.fullscreen = false;
    return state;
}

void panAttention(AttentionState& state, int d_row, int d_col) {
    const int n = static_cast<int>(state.tokens.size());
    const int max_offset = max(0, n - state.window);

    state.row_offset = clamp(state.row_offset + d_row, 0, max_offset);
    state.col_offset = clamp(state.col_offset + d_col, 0, max_offset);
}

void adjustContrast(AttentionState& state, double delta) {
    state.contrast += delta;
    if (state.contrast < 0.25) state.contrast = 0.25;
    if (state.contrast > 3.0) state.contrast = 3.0;
}

void toggleFullscreen(AttentionState& state) {
    state.fullscreen = !state.fullscreen;
}

void clampAttentionView(AttentionState& state) {
    const int n = static_cast<int>(state.tokens.size());
    const int max_offset = max(0, n - state.window);

    if (state.row_offset < 0) state.row_offset = 0;
    if (state.col_offset < 0) state.col_offset = 0;
    if (state.row_offset > max_offset) state.row_offset = max_offset;
    if (state.col_offset > max_offset) state.col_offset = max_offset;
}