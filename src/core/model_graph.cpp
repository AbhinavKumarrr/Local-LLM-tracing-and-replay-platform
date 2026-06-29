#include "llmscope/model_graph.hpp"

namespace llmscope {

const char* to_string(ModuleKind kind) {
    switch (kind) {
        case ModuleKind::Embedding:       return "embedding";
        case ModuleKind::LayerNorm:       return "layernorm";
        case ModuleKind::Attention:       return "attention";
        case ModuleKind::AttentionScores: return "attn_scores";
        case ModuleKind::MLP:             return "mlp";
        case ModuleKind::Residual:        return "residual";
        case ModuleKind::LogitsHead:      return "logits";
        case ModuleKind::Other:           return "other";
    }
    return "other";
}

namespace {
void flatten_into(const GraphNode& node,
                  std::vector<const GraphNode*>& out) {
    out.push_back(&node);
    for (const GraphNode& child : node.children) {
        flatten_into(child, out);
    }
}
}  // namespace

std::vector<const GraphNode*> ModelGraph::flatten() const {
    std::vector<const GraphNode*> out;
    flatten_into(root, out);
    return out;
}

ModelGraph ModelGraph::build_reference(const std::string& model_name,
                                       int n_layers) {
    ModelGraph g;
    g.root.name = model_name;
    g.root.path = model_name;
    g.root.kind = ModuleKind::Other;
    g.root.depth = 0;

    GraphNode embed;
    embed.name = "embed_tokens";
    embed.path = "embed_tokens";
    embed.kind = ModuleKind::Embedding;
    embed.depth = 1;
    g.root.children.push_back(embed);

    GraphNode layers;
    layers.name = "layers";
    layers.path = "layers";
    layers.kind = ModuleKind::Other;
    layers.depth = 1;

    for (int i = 0; i < n_layers; ++i) {
        const std::string base = "layers." + std::to_string(i);
        GraphNode block;
        block.name = base;
        block.path = base;
        block.kind = ModuleKind::Other;
        block.layer_index = i;
        block.depth = 2;

        const char* sub_names[] = {"input_layernorm", "attn",
                                   "post_attention_layernorm", "mlp"};
        const ModuleKind sub_kinds[] = {ModuleKind::LayerNorm,
                                        ModuleKind::Attention,
                                        ModuleKind::LayerNorm, ModuleKind::MLP};

        for (int s = 0; s < 4; ++s) {
            GraphNode sub;
            sub.name = sub_names[s];
            sub.path = base + "." + sub_names[s];
            sub.kind = sub_kinds[s];
            sub.layer_index = i;
            sub.depth = 3;
            block.children.push_back(sub);
        }
        layers.children.push_back(block);
    }

    g.root.children.push_back(layers);

    GraphNode norm;
    norm.name = "norm";
    norm.path = "norm";
    norm.kind = ModuleKind::LayerNorm;
    norm.depth = 1;
    g.root.children.push_back(norm);

    GraphNode head;
    head.name = "lm_head";
    head.path = "lm_head";
    head.kind = ModuleKind::LogitsHead;
    head.depth = 1;
    g.root.children.push_back(head);

    return g;
}

}  // namespace llmscope