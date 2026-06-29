#pragma once
#include <cstddef>
#include <vector>

namespace llmscope {

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t cap = 256)
        : buffer_(cap), cap_(cap) {}

    void push(T value) {
        if (cap_ == 0) return;
        if (size_ < cap_) {
            buffer_[(start_ + size_) % cap_] = std::move(value);
            ++size_;
        } else {
            buffer_[start_] = std::move(value);
            start_ = (start_ + 1) % cap_;
            ++dropped_;
        }
    }

    void clear() {
        start_ = 0;
        size_ = 0;
        dropped_ = 0;
    }

    std::vector<T> snapshot() const {
        std::vector<T> out;
        out.reserve(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            out.push_back(buffer_[(start_ + i) % cap_]);
        }
        return out;
    }

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return cap_; }
    std::size_t dropped() const { return dropped_; }

private:
    std::vector<T> buffer_;
    std::size_t cap_ = 0;
    std::size_t start_ = 0;
    std::size_t size_ = 0;
    std::size_t dropped_ = 0;
};

}  // namespace llmscope