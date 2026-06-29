#pragma once
#include <string>
#include <vector>
#include "event.hpp"

namespace llmscope {

const char* to_string(ModuleKind kind);

struct GraphNode {
    std::string name;
    std::string path;
    ModuleKind kind = ModuleKind::Other;
    int layer_index = -1;
    int depth = 0;
    std::vector<GraphNode> children;
};

struct ModelGraph {
    GraphNode root;

    std::vector<const GraphNode*> flatten() const;
    static ModelGraph build_reference(const std::string& model_name, int n_layers);
};

}  // namespace llmscope