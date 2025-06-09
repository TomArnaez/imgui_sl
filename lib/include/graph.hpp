#pragma once

#include <utility/slot_map.hpp>
#include <vulkan/vulkan.hpp>

#include <expected>
#include <functional>
#include <map>
#include <ranges>
#include <set>
#include <string_view>

namespace vkengine {

struct resource_access {
  vk::PipelineStageFlags2 stage_mask = {};
  vk::AccessFlags2        access_mask = {};
  vk::ImageLayout         image_layout = {};
  uint32_t                queue_family_index = vk::QueueFamilyIgnored;

  bool contains_write() const noexcept {
    constexpr auto w =
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eShaderWrite |
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite;
    return static_cast<bool>(access_mask & w);
  }

  bool contains_read() const noexcept {
    constexpr auto r =
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eShaderRead |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eHostRead;
    return static_cast<bool>(access_mask & r);
  }
};

struct buffer_state {
  resource_access access = {};
  vk::Buffer      buffer = nullptr;
};

struct image_state {
  resource_access access = {};
  vk::ImageLayout image_layout = vk::ImageLayout::eUndefined;
  vk::Image       image = nullptr;
};

enum class object_type : uint32_t { BUFFER = 0, IMAGE = 1 };

template <class R> struct resource_tag;

template <>
struct resource_tag<vk::Buffer>
    : std::integral_constant<object_type, object_type::BUFFER> {};

template <>
struct resource_tag<vk::Image>
    : std::integral_constant<object_type, object_type::IMAGE> {};

template <class R>
concept Resource = requires { resource_tag<R>::value; };

template<typename Res = void>
    requires (std::same_as<Res, void> || Resource<Res>)
class id {
public:
  explicit constexpr id(slot_id s) noexcept : slot_(s) {}

  constexpr std::uint32_t index() const noexcept { return slot_.index(); }

  template <Resource O> constexpr bool is() const noexcept {
    if constexpr (!std::same_as<Res, void>)
      return std::same_as<Res, O>;
    else
      return type() == resource_tag<O>::value;
  }

  constexpr object_type type() const noexcept {
    if constexpr (!std::same_as<Res, void>)
      return resource_tag<Res>::value;
    else
      return decode_runtime_tag();
  }

private:
  slot_id slot_;

  constexpr object_type decode_runtime_tag() const noexcept {
    switch (slot_.tag() & 0b1111) {
    case static_cast<std::uint32_t>(object_type::BUFFER):
      return object_type::BUFFER;
    case static_cast<std::uint32_t>(object_type::IMAGE):
      return object_type::IMAGE;
    default:
      std::unreachable();
    }
  }
};

using untyped_id = id<>;

class resource_manager {
public:
  resource_manager(vk::Device device) : device_(device) {}

  [[nodiscard]]
  auto buffer(slot_map<buffer_state>::id id) const {
    return buffer_state_.get(id);
  }

  [[nodiscard]]
  decltype(auto) buffer_unchecked(this auto &&self, std::uint32_t index) {
    return self.buffer_state_.get_unchecked(index);
  }

  [[nodiscard]]
  auto image(slot_map<image_state>::id id) const {
    return image_state_.get(id);
  }

  [[nodiscard]]
  auto add_buffer(vk::Buffer buffer) {
    return buffer_state_.emplace(buffer_state{.buffer = buffer});
  }

  [[nodiscard]]
  auto add_image(vk::Image image) {
    return image_state_.emplace(image_state{.image = image});
  }

private:
  vk::Device device_;

  slot_map<buffer_state> buffer_state_;
  slot_map<image_state>  image_state_;
};

struct task_node;

using node_index = uint32_t;

struct task_node {
  std::map<untyped_id, resource_access>              resource_acceses;
  std::vector<node_index>                          in_edges;
  std::vector<node_index>                          out_edges;
  uint32_t                                         queue_family_index = 0;
  std::move_only_function<void(vk::CommandBuffer)> execute;
};

using node_id = slot_id;
class executable_task_graph;

class task_graph {
public:
  [[nodiscard]]
  node_id add_node(task_node &&node) {
    return task_nodes_.emplace(std::forward<task_node>(node));
  }

  [[nodiscard]]
  std::expected<node_id, std::string> remove_node(node_id id) {

    /*---- 1. look up the node we are about to delete ----*/
    auto node_exp = task_nodes_.get(id);
    if (!node_exp)
      return std::unexpected("node not found");

    task_node &node = *node_exp.value();

    /*---- 2. detach us from every OUT edge ----*/
    for (node_index out_index : node.out_edges) {
      auto &out = task_nodes_.get_unchecked(out_index);
      std::erase(out.in_edges, out_index);
    }

    /*---- 3. detach us from every IN edge ----*/
    for (node_index in_index : node.in_edges) {
      auto &in = task_nodes_.get_unchecked(in_index);
      std::erase(in.out_edges, in_index);
    }

    /*---- 4. finally remove the node itself ----*/
    if (auto res = task_nodes_.remove(id); !res)
      return std::unexpected("internal slot_map error");

    return id;
  }

  [[nodiscard]]
  std::expected<std::monostate, std::string> add_edge(node_id from,
                                                      node_id to) {
    auto &from_node = task_nodes_.get_unchecked(from.index());
    auto &to_node = task_nodes_.get_unchecked(to.index());

    // We won't check for cycles here, just duplicate edges
    auto &from_out_edges = from_node.out_edges;
    if (std::ranges::find(from_out_edges, to.index()) != from_out_edges.end())
      return std::unexpected("edge already exists");

    return {};
  }

  [[nodiscard]]
  std::expected<std::monostate, std::string> remove_edge(node_id from,
                                                         node_id to) {
    auto &from_node = task_nodes_.get_unchecked(from.index());
    auto &to_node = task_nodes_.get_unchecked(to.index());

    std::erase(from_node.out_edges, to.index());
    std::erase(to_node.in_edges, from.index());
  }

  [[nodiscard]]
  decltype(auto) node(this auto &&self, node_id id) {
    return self.task_nodes_.get(id);
  }

  [[nodiscard]] decltype(auto) node_unchecked(this auto &&self,
                                              node_index  index) {
    return self.task_nodes_.get_unchecked(index);
  }

  [[nodiscard]]
  auto nodes() const {
    return task_nodes_.entries();
  }

  enum compile_error { unconnected };

  [[nodiscard]]
  std::expected<executable_task_graph, compile_error> compile() && noexcept;

  [[nodiscard]]
  size_t size() const noexcept {
    return task_nodes_.size();
  }

private:
  slot_map<task_node> task_nodes_;

  std::expected<std::vector<node_index>, compile_error>
  topological_sort() const noexcept;
};

struct executable_task_graph {
  task_graph graph;
};

} // namespace vkengine