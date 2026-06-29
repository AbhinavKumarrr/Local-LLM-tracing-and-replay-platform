#include "llmscope/tokenizer.hpp"

#include <cctype>

namespace llmscope {

int Tokenizer::intern(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    const int id = static_cast<int>(vocab_.size());
    token_to_id_.emplace(token, id);
    vocab_.push_back(token);
    return id;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    std::string current;

    auto flush = [&]() {
        if (!current.empty()) {
            ids.push_back(intern(current));
            current.clear();
        }
    };

    for (char c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            current.push_back(static_cast<char>(std::tolower(uc)));
        } else if (std::isspace(uc)) {
            flush();
        } else {
            flush();
            ids.push_back(intern(std::string(1, c)));
        }
    }
    flush();
    return ids;
}

const std::string& Tokenizer::decode(int id) const {
    static const std::string kUnknown = "<unk>";
    if (id < 0 || id >= static_cast<int>(vocab_.size())) {
        return kUnknown;
    }
    return vocab_[static_cast<std::size_t>(id)];
}

}  // namespace llmscope