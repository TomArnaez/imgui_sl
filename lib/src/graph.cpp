#include <generator>
#include <graph.hpp>
#include <range/v3/view/concat.hpp>
#include <ranges>

namespace {

using namespace vkengine;

struct memory_barrier {
  vk::PipelineStageFlags2 src_stage_mask;
  vk::AccessFlags2        src_access_mask;
  vk::PipelineStageFlags2 dst_stage_mask;
  vk::AccessFlags2        dst_access_mask;
  vk::ImageLayout         old_layout;
  vk::ImageLayout         new_layout;
  uint32_t                src_queue_family_index;
  uint32_t                dst_queue_family_index;
  untyped_id              resource;
};

using semaphore_index = uint32_t;

struct submission_state {
  node_index                  first_node_index;
  node_index                  last_node_index;
  std::vector<memory_barrier> initial_barriers;

  submission_state(node_index node_index)
      : first_node_index(node_index), last_node_index(node_index) {}
};

struct node_state {
  size_t                       submission_index;
  std::vector<memory_barrier>  start_barriers;
  std::vector<memory_barrier>  end_barriers;
  std::vector<semaphore_index> semaphore_waits;
  std::vector<semaphore_index> semaphore_signals;
};

struct executable_graph_builder {
  std::vector<submission_state> submissions_;
  std::vector<node_state>       node_states_;
  std::vector<resource_access>  prev_accesses_;
  std::vector<node_index>       prev_node_indices_;
  uint32_t                      semaphore_count_ = 0;

  executable_graph_builder(uint32_t node_capacity, uint32_t resource_capacity)
      : node_states_(node_capacity), prev_accesses_(resource_capacity),
        prev_node_indices_(0, resource_capacity) {}

  void add_resource_access(node_index node_index, untyped_id id,
                           const resource_access &access) {
    auto &prev_access = prev_accesses_[id.index()];
    auto &prev_node_index = prev_node_indices_[id.index()];
    bool  barriered = true;

    if (!prev_access.stage_mask) {
      if (id.is<vk::Image>())
        initial_image_layout_transition(id, access);
      else if (access.contains_read())
        initial_memory_barrier(id, access);
    } else if (prev_access.queue_family_index != access.queue_family_index) {
      // Assumes exclusive access
      queue_family_ownership_release(prev_node_index, id, access);
      queue_family_ownership_acquire(node_index, id, access);
    } else if (prev_access.image_layout != access.image_layout)
      image_layout_transition(node_index, id, access);
    else if (prev_access.contains_write())
      add_memory_barrier(node_index, id, access);
    else if (access.contains_write())
      execution_barrier(node_index, id, access);
    else
      barriered = false;

    if (barriered) {
      prev_access = access;
      prev_node_index = id.index();
    } else {
      prev_access.access_mask |= access.access_mask;
      prev_access.stage_mask |= access.stage_mask;
      auto node_barriers_view =
          node_states_ | std::views::transform(&node_state::start_barriers) |
          std::views::join;

      memory_barrier &prev_barrier =
          ranges::views::concat(node_barriers_view,
                                submissions_.back().initial_barriers)
              .front();
      prev_barrier.dst_access_mask |= access.access_mask;
      prev_barrier.dst_stage_mask |= access.stage_mask;
    }
  }

  void buffer_access(uint32_t buffer_id, node_index node_index,
                     const resource_access &access) {
    auto &node = node_states_[node_index];
  }

  void semaphore_signal(node_index signal_node_index,
                        node_index wait_node_index) {
    node_states_[wait_node_index].semaphore_waits.push_back(semaphore_count_);
    node_states_[signal_node_index].semaphore_signals.push_back(
        semaphore_count_);
    semaphore_count_++;
  }

private:
  void initial_image_layout_transition(untyped_id             id,
                                       const resource_access &access) {
    initial_memory_barrier(id, access);
  }

  void initial_memory_barrier(untyped_id id, const resource_access &access) {
    auto &submission = submissions_.back();

    submission.initial_barriers.emplace_back(
        memory_barrier{.src_stage_mask = {},
                       .src_access_mask = {},
                       .dst_stage_mask = access.stage_mask,
                       .dst_access_mask = access.access_mask,
                       .old_layout = vk::ImageLayout::eUndefined,
                       .new_layout = access.image_layout,
                       .src_queue_family_index = vk::QueueFamilyIgnored,
                       .dst_queue_family_index = vk::QueueFamilyIgnored,
                       .resource = id});
  }

  void image_layout_transition(node_index node_index, untyped_id id,
                               const resource_access &access) {
    assert(prev_accesses_[id.index()].image_layout != access.image_layout);

    add_memory_barrier(node_index, id, access);
  }

  void queue_family_ownership_release(node_index node_index, untyped_id id,
                                      const resource_access &access) {
    const resource_access &prev_access = prev_accesses_[id.index()];

    resource_access src = prev_access;
    resource_access dst = {.image_layout = access.image_layout,
                           .queue_family_index = access.queue_family_index};

    if (prev_access.contains_write())
      src.access_mask = {};

    memory_barrier_inner(node_index, id, src, dst, false);
  }

  void queue_family_ownership_acquire(node_index node_index, untyped_id id,
                                      const resource_access &access) {
    const resource_access &prev_access = prev_accesses_[id.index()];
    resource_access        src = {.image_layout = prev_access.image_layout,
                                  .queue_family_index =
                                      prev_access.queue_family_index};
    const resource_access &dst = access;

    assert(src.queue_family_index != dst.queue_family_index);

    memory_barrier_inner(node_index, id, src, dst, false);
  }

  void execution_barrier(node_index node_index, untyped_id id,
                         const resource_access &access) {
    const resource_access &prev_access = prev_accesses_[id.index()];

    assert(prev_access.image_layout == access.image_layout);

    resource_access src = prev_access;
    src.access_mask = {};
    src.queue_family_index = vk::QueueFamilyIgnored;

    resource_access dst = access;
    dst.access_mask = {};

    memory_barrier_inner(node_index, id, src, dst, false);
  }

  void add_memory_barrier(node_index node_index, untyped_id id,
                          const resource_access &access) {
    const resource_access &prev_access = prev_accesses_[id.index()];

    resource_access src = prev_access;
    src.queue_family_index = vk::QueueFamilyIgnored;

    resource_access dst = access;
    dst.queue_family_index = vk::QueueFamilyIgnored;

    memory_barrier_inner(node_index, id, src, dst, false);
  }

  void memory_barrier_inner(node_index node_index, untyped_id id,
                            const resource_access &src,
                            const resource_access &dst, bool is_end_barrier) {
    auto &node_state = node_states_[node_index];
    auto &barriers =
        is_end_barrier ? node_state.start_barriers : node_state.end_barriers;

    barriers.emplace_back(
        memory_barrier{.src_stage_mask = src.stage_mask,
                       .src_access_mask = src.access_mask,
                       .dst_stage_mask = dst.stage_mask,
                       .dst_access_mask = dst.access_mask,
                       .src_queue_family_index = src.queue_family_index,
                       .dst_queue_family_index = dst.queue_family_index,
                       .resource = id});
  }
};

void build_ir(executable_graph_builder      &builder,
              const std::vector<node_index> &topological_order,
              const slot_map<task_node>     &task_nodes) {
  uint32_t prev_queue_family_index = vk::QueueFamilyIgnored;

  for (auto node_index : topological_order) {
    auto &node = task_nodes.get_unchecked(node_index);

    uint32_t queue_family_index = node.queue_family_index;

    for (auto out_node_index : node.out_edges) {
      auto &out_node = task_nodes.get_unchecked(out_node_index);

      if (queue_family_index != out_node.queue_family_index)
        builder.semaphore_signal(node_index, out_node_index);

      if (prev_queue_family_index != queue_family_index)
        builder.submissions_.emplace_back(submission_state(node_index));

      auto submission_index = builder.submissions_.size() - 1;
      builder.node_states_[submission_index].submission_index =
          submission_index;

      for (auto const &[id, access] : node.resource_acceses) {
        resource_access access = access;
        access.queue_family_index = queue_family_index;

        builder.add_resource_access(node_index, id, access);
      }

      builder.submissions_.back().last_node_index = node_index;

      prev_queue_family_index = queue_family_index;
    }
  }
}



} // namespace

namespace vkengine {

std::expected<executable_task_graph, task_graph::compile_error>
task_graph::compile() && noexcept {
  executable_graph_builder builder(task_nodes_.capacity(), 1024);

  auto topological_order = topological_sort();
  build_ir(builder, *topological_order, task_nodes_);

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