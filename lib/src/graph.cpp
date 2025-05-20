#include <graph.hpp>

namespace vkengine {

struct memory_barrier {
  vk::PipelineStageFlags2 src_stage_mask;
  vk::AccessFlags2        src_access_mask;
  vk::PipelineStageFlags2 dst_stage_mask;
  vk::AccessFlags2        dst_access_mask;
  uint32_t                src_queue_family_index;
  uint32_t                dst_queue_family_index;
  uint32_t                resource_id;
};

using semaphore_index = uint32_t;

struct node_state {
  std::vector<memory_barrier>  barriers;
  std::vector<semaphore_index> semaphore_waits;
  std::vector<semaphore_index> semaphore_signals;
};

struct executable_graph_builder {
  resource_manager       &resource_manager_;
  std::vector<node_state> node_states_;
  uint32_t                semaphore_count_ = 0;

  executable_graph_builder(uint32_t          node_capacity,
                           resource_manager &resource_manager)
      : node_states_(node_capacity), resource_manager_(resource_manager) {}

  void buffer_access(uint32_t buffer_id, node_index node_index,
                     const resource_access &access) {
    auto &node = node_states_[node_index];

    auto &buffer_state = resource_manager_.buffer_unchecked(buffer_id);

    if (access.queue_family_index != buffer_state.access.queue_family_index) {
      queue_family_ownership_acquire(node_index, buffer_id, access);
      queue_family_ownership_release(node_index, buffer_id, access);
    }
  }

  void semaphore_signal(node_index signal_node_index,
                        node_index wait_node_index) {
    node_states_[wait_node_index].semaphore_waits.push_back(semaphore_count_);
    node_states_[signal_node_index].semaphore_signals.push_back(
        semaphore_count_);
    semaphore_count_++;
  }

private:
  void queue_family_ownership_release(node_index node_index, uint32_t buffer_id,
                                      const resource_access &access) {
    auto &buffer_state = resource_manager_.buffer_unchecked(buffer_id);

    assert(buffer_state.access.queue_family_index != access.queue_family_index);

    resource_access dst = {.stage_mask = {},
                           .access_mask = {},
                           .queue_family_index = access.queue_family_index};
  }

  void queue_family_ownership_acquire(node_index node_index, uint32_t buffer_id,
                                      const resource_access &access) {
    auto &buffer_state = resource_manager_.buffer_unchecked(buffer_id);

    assert(buffer_state.access.queue_family_index != access.queue_family_index);

    resource_access src = {.stage_mask = {},
                           .access_mask = {},
                           .queue_family_index = access.queue_family_index};
  }

  void add_memory_barrier(node_index node_index, uint32_t buffer_id,
                          const resource_access &src,
                          const resource_access &dst) {
    auto &node_state = node_states_[node_index];

    node_state.barriers.emplace_back(
        memory_barrier{.src_stage_mask = src.stage_mask,
                       .src_access_mask = src.access_mask,
                       .dst_stage_mask = dst.stage_mask,
                       .dst_access_mask = dst.access_mask,
                       .src_queue_family_index = src.queue_family_index,
                       .dst_queue_family_index = dst.queue_family_index});
  }
};

std::expected<executable_task_graph, task_graph::compile_error>
task_graph::compile() && noexcept {
  executable_graph_builder builder(task_nodes_.capacity(),
                                   resource_manager_.get());

  auto topological_order = topological_sort();
  if (!topological_order)
    return std::unexpected(topological_order.error());

  for (auto node_index : *topological_order) {
    auto &node = task_nodes_.get_unchecked(node_index);

    uint32_t curr_queue_family_index = node.queue_family_index;

    for (auto out_node_index : node.out_edges) {
      auto &out_node = task_nodes_.get_unchecked(out_node_index);

      if (curr_queue_family_index != out_node.queue_family_index)
        builder.semaphore_signal(curr_queue_family_index,
                                 out_node.queue_family_index);
    }
  }

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