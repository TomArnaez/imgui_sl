#include <graph.hpp>

namespace vkengine {

struct memory_barrier {
  vk::PipelineStageFlagBits2 src_stage_mask;
  vk::AccessFlagBits2        src_access_mask;
  vk::PipelineStageFlagBits2 dst_stage_mask;
  vk::AccessFlagBits2        dst_access_mask;
  uint32_t                   src_queue_family_index;
  uint32_t                   dst_queue_family_index;
  uint32_t                   resource_id;
};

struct node_state {
  std::vector<memory_barrier> start_barriers;
};

struct executable_graph_builder {
  resource_manager &resource_manager;
};

std::expected<executable_task_graph, task_graph::compile_error>
task_graph::compile() && noexcept {
  auto sort_result = topological_sort();
  if (!sort_result)
    return std::unexpected(sort_result.error());

  return executable_task_graph{.graph = std::move(*this)};
}

std::expected<std::vector<node_index>, task_graph::compile_error>
task_graph::topological_sort() const noexcept {
  std::vector<node_index>                  queue(size());
  std::unordered_map<node_index, uint32_t> indeg;

  for (auto [id, node] : nodes()) {
    if (node.in_edges.empty())
      queue.push_back(id.value);
    indeg[id.value] = node.in_edges.size();
  }

  std::vector<node_index> order;

  while (!queue.empty()) {
    node_index index = queue.back();

    queue.pop_back();
    order.push_back(index);

    auto &node = node_unchecked(index);

    for (auto v : node.out_edges) {
      auto &out_node = node_unchecked(v);

      if (--indeg[v] == 0)
        queue.push_back(v);
    }
  }

  if (order.size() != size())
    return std::unexpected(compile_error::unconnected);

  return order;
}

} // namespace vkengine