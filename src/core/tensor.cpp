#include "llmscope/tensor.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace llmscope {

namespace {
std::string join_shape(const std::vector<int64_t>& shape) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ", ";
        os << shape[i];
    }
    os << ']';
    return os.str();
}
}  // namespace

Tensor::Tensor(std::vector<int64_t> shp, float fill) : shape(std::move(shp)) {
    data.assign(numel(), fill);
}

std::size_t Tensor::numel() const {
    if (shape.empty()) return 0;
    std::size_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) return 0;
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::string Tensor::shape_string() const { return join_shape(shape); }

TensorStats TensorStats::compute(const Tensor& t, float zero_eps) {
    return compute(t.shape, t.data.data(), t.data.size(), t.dtype, zero_eps);
}

TensorStats TensorStats::compute(const std::vector<int64_t>& shape,
                                 const float* data,
                                 std::size_t count,
                                 const std::string& dtype,
                                 float zero_eps) {
    TensorStats s;
    s.shape = shape;
    s.dtype = dtype;
    s.count = count;

    if (count == 0 || data == nullptr) {
        return s;
    }

    double sum = 0.0;
    double sq_sum = 0.0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    float abs_mx = 0.0f;
    std::size_t zeros = 0;

    for (std::size_t i = 0; i < count; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            s.has_nan = true;
            continue;
        }
        if (std::isinf(v)) {
            s.has_inf = true;
            continue;
        }
        sum += v;
        sq_sum += static_cast<double>(v) * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        const float av = std::fabs(v);
        if (av > abs_mx) abs_mx = av;
        if (av < zero_eps) ++zeros;
    }

    s.mean = static_cast<float>(sum / static_cast<double>(count));
    s.min = std::isinf(mn) ? 0.0f : mn;
    s.max = std::isinf(mx) ? 0.0f : mx;
    s.abs_max = abs_mx;
    s.l2_norm = static_cast<float>(std::sqrt(sq_sum));
    s.sparsity = static_cast<double>(zeros) / static_cast<double>(count);
    return s;
}

ScopedTimer::ScopedTimer() { reset(); }

void ScopedTimer::reset() {
    using namespace std::chrono;
    start_ns_ = static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

double ScopedTimer::elapsed_ms() const {
    using namespace std::chrono;
    const uint64_t now = static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
    return static_cast<double>(now - start_ns_) / 1'000'000.0;
}

}  // namespace llmscope