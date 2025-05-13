#pragma once

#include <vulkan/vulkan.hpp>
#include <utility/slot_map.hpp>

#include <expected>
#include <functional>
#include <string_view>
#include <set>

struct base_resource_state {
	vk::PipelineStageFlags2 stage = {};
	vk::AccessFlags2		access = {};
	uint32_t				queue_family = vk::QueueFamilyIgnored;
};

struct buffer_state {
	base_resource_state	resource_state = {};
	vk::Buffer			buffer = nullptr;
};

struct image_state {
	base_resource_state	resource_state = {};
	vk::ImageLayout		image_layout = vk::ImageLayout::eUndefined;
	vk::Image			image = nullptr;
};

class resource_manager {
public:
	resource_manager(vk::Device device)
		: device_(device) {
	}

	auto buffer(slot_map<buffer_state>::id id) const {
		return buffer_state_.get(id);
	}

	auto image(slot_map<image_state>::id id) const {
		return image_state_.get(id);
	}

	auto add_buffer(vk::Buffer buffer) {
		return buffer_state_.emplace(buffer_state{
			.buffer = buffer
			});
	}

	auto add_image(vk::Image image) {
		return image_state_.emplace(image_state{
			.image = image
			});
	}

private:
	vk::Device				device_;

	slot_map<buffer_state>	buffer_state_;
	slot_map<image_state>	image_state_;
};

struct task_node;

using node_index = uint32_t;

struct task_node {
	std::vector<image_state>							image_accesses;
	std::vector<buffer_state>							buffer_accesses;
	std::vector<node_index>								in_edges;
	std::vector<node_index>								out_edges;
	uint32_t											queue_family_index = 0;
	std::move_only_function<void(vk::CommandBuffer)>	execute;
};

class task_graph {
	using node_id = slot_map<task_node>::id;
public:
	[[nodiscard]]
	std::expected<node_id, std::string>
	add_node(task_node&& node) {
		auto id = task_nodes_.emplace(std::forward<task_node>(node));
		if (!id) return std::unexpected("internal slot_map error");
		return id.value();
	}

    [[nodiscard]]
    std::expected<node_id, std::string> 
	remove_node(node_id id) {

        /*–––– 1. look up the node we are about to delete ––––*/
        auto node_exp = task_nodes_.get(id);
        if (!node_exp) return std::unexpected("node not found");

        task_node& node = *node_exp.value();

        /*–––– 2. detach us from every OUT edge ––––*/
        for (node_index out_index : node.out_edges) {
            auto out_exp = task_nodes_.get(node_id(out_index));
            if (!out_exp)
                return std::unexpected("dangling out-edge");

            std::erase(out_exp.value()->in_edges, out_index);
        }

        /*–––– 3. detach us from every IN edge ––––*/
        for (node_index in_index : node.in_edges) {
            auto in_exp = task_nodes_.get(node_id(in_index));
            if (!in_exp) return std::unexpected("dangling in-edge");

            std::erase(in_exp.value()->out_edges, in_index);
        }

        /*–––– 4. finally remove the node itself ––––*/
        if (auto res = task_nodes_.remove(id); !res)
            return std::unexpected("internal slot_map error");

        return id;
    }

	[[nodiscard]]
	std::expected<std::monostate, std::string> 
	add_edge(node_id from, node_id to) {
		auto from_node_res = task_nodes_.get(from);
		if (!from_node_res)
			return std::unexpected("from node not found");

		auto to_node_res = task_nodes_.get(to);
		if (!to_node_res)
			return std::unexpected("to node not found");

		// We won't check for cycles here, just duplicate edges
		auto& from_out_edges = from_node_res.value()->out_edges;
		if (std::ranges::find(from_out_edges, to.value) != from_out_edges.end())
			return std::unexpected("edge already exists");

		return {};
	}

	[[nodiscard]]
	std::expected<std::monostate, std::string> 
	remove_edge(node_id from, node_id to) {
		auto from_node_res = task_nodes_.get(from);
		if (!from_node_res) return std::unexpected("from node not found");

		auto to_node_res = task_nodes_.get(to);
		if (!to_node_res) return std::unexpected("to node not found");

		std::erase(from_node_res.value()->out_edges, to.value);
		std::erase(to_node_res.value()->in_edges, from.value);
	}

	[[nodiscard]]
	auto nodes() const {
		return task_nodes_.values();
	}

	[[nodiscard]] 
	size_t size() const noexcept {
		return task_nodes_.size();
	}

private:
    std::reference_wrapper<resource_manager> resource_manager_;
    slot_map<task_node>                      task_nodes_;
};

// Kahn's algorihm for topological sorting: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
std::expected<std::vector<node_index>, std::string>
compile_graph(task_graph& graph) {

	std::vector<uint32_t> queue(std::from_range, graph.nodes()) | std::views::filter([&](auto& node) { return node->in_edges.empty(); });


	while (!queue.empty()) {
/*		auto u = queue.back();
		queue.pop_back();
		order.emplace_back((u)*/;
		//for (auto v : graph.nodes()[u].out_edges)
		//	if (--indeg[v] == 0) queue.push_back(v);
	}

	if (order.size() != graph.size())
		return std::unexpected("Cycle detected");

	return order;
}
