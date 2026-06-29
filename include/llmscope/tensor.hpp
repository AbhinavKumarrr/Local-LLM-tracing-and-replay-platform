#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llmscope {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
    std::string dtype = "float32";

    Tensor() = default;
    Tensor(std::vector<int64_t> shp, float fill = 0.0f);

    std::size_t numel() const;
    std::string shape_string() const;
};

struct TensorStats {
    std::vector<int64_t> shape;
    std::string dtype = "float32";
    std::size_t count = 0;

    float mean = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    float abs_max = 0.0f;
    float l2_norm = 0.0f;
    double sparsity = 0.0;
    bool has_nan = false;
    bool has_inf = false;

    std::string shape_string() const;

    static TensorStats compute(const Tensor& t, float zero_eps = 1e-6f);
    static TensorStats compute(const std::vector<int64_t>& shape,
                               const float* data,
                               std::size_t count,
                               const std::string& dtype,
                               float zero_eps = 1e-6f);
};

class ScopedTimer {
public:
    ScopedTimer();
    void reset();
    double elapsed_ms() const;

private:
    uint64_t start_ns_;
};

}  // namespace llmscope