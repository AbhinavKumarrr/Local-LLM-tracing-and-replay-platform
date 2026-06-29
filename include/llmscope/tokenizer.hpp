#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace llmscope {

class Tokenizer {
public:
    std::vector<int> encode(const std::string& text);
    const std::string& decode(int id) const;

private:
    int intern(const std::string& token);

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> vocab_;
};

}  // namespace llmscope