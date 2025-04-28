#pragma once

#include <detailed_exception.hpp>
#include <unordered_map>
#include <functional>
#include <variant>
#include <queue>

namespace vkengine {

struct buffer_usage {
	vk::Buffer				buffer;
	vk::AccessFlags2		access;
	vk::PipelineStageFlags2 stages;
};

template<typename Op>
concept operation = requires(const Op& op, vk::CommandBuffer cb) {
	{ op.usages() }		-> std::ranges::forward_range;
	{ op.record(cb) }	-> std::same_as<void>;
};

struct compiled_graph {
	struct step {
		std::vector<vk::BufferMemoryBarrier2>	buffer_memory_barriers;
		std::function<void(vk::CommandBuffer)>	record;
	};

	std::vector<step> steps;
};

constexpr vk::AccessFlags2 WRITE_MASK =
	vk::AccessFlagBits2::eMemoryWrite
	| vk::AccessFlagBits2::eShaderWrite
	| vk::AccessFlagBits2::eTransferWrite
	| vk::AccessFlagBits2::eHostWrite;

constexpr bool is_write(vk::AccessFlags2 a) noexcept {
	return static_cast<bool>(a & WRITE_MASK);
}

struct buffer_hash {
	std::size_t operator()(const vk::Buffer& b) const noexcept {
		return std::hash<VkBuffer>{}(static_cast<VkBuffer>(b));
	}
};

template<operation... ops>
using op_variant = std::variant<ops...>;

template<operation... ops>
compiled_graph compile(const std::vector<op_variant<ops...>>& nodes) {
	using variant = std::variant<ops...>;
	using last_key = vk::Buffer;

	struct last_use {
		vk::AccessFlags2        access		= {};
		vk::PipelineStageFlags2 stages		= {};
		std::size_t             node_index	= 0;					// who produced it
	};

	struct tmp_node {
		std::vector<vk::BufferMemoryBarrier2>	buffer_memory_barriers;			// barriers attached here
		std::function<void(vk::CommandBuffer)>	record;
		std::vector<std::size_t>				out_edges;		// DAG edges for topo sort
	};

	std::vector<tmp_node> tmp(nodes.size());
	std::unordered_map<vk::Buffer, last_use,
		buffer_hash,
		std::equal_to<>>
		last;

	for (std::size_t i = 0; i < nodes.size(); ++i) {
		const variant& node = nodes[i];

		std::visit([&](const auto& op) { 
			tmp[i].record = [op](vk::CommandBuffer cb) { op.record(cb); };

			for (const buffer_usage& usage : op.usages()) {
				last_key key = usage.buffer;

				if (auto it = last.find(key); it != last.end()) {
					last_use& prev = it->second;

					if (prev.node_index != i) {
						bool need_order = is_write(prev.access) || is_write(usage.access);
						if (need_order) {
							auto barrier = vk::BufferMemoryBarrier2()
								.setSrcStageMask(prev.stages)
								.setDstStageMask(usage.stages)
								.setSrcAccessMask(prev.access)
								.setDstAccessMask(usage.access)
								.setBuffer(usage.buffer)
								.setOffset(0)
								.setSize(VK_WHOLE_SIZE);

							tmp[i].buffer_memory_barriers.emplace_back(barrier);

							tmp[prev.node_index].out_edges.push_back(i);
						}
					}

					// merge – don’t overwrite (so both usages are visible to later nodes)
					prev.access |= usage.access;
					prev.stages |= usage.stages;
					prev.node_index = i;			// "last" writer/reader is still this node
				} else
					last[key] = { usage.access, usage.stages, i };
			}
		}, node);
	}

	std::vector<std::size_t> indeg(tmp.size(), 0);
	for (const auto& n : tmp)
		for (auto v : n.out_edges) ++indeg[v];

	std::vector<std::size_t> queue;
	queue.reserve(tmp.size());
	for (std::size_t i = 0; i < indeg.size(); ++i)
		if (!indeg[i]) queue.push_back(i);

	std::vector<std::size_t> order;
	order.reserve(tmp.size());

	while (!queue.empty()) {
		auto u = queue.back();
		queue.pop_back();
		order.push_back(u);
		for (auto v : tmp[u].out_edges)
			if (--indeg[v] == 0) queue.push_back(v);
	}

	if (order.size() != tmp.size())
		throw detailed_exception("graph: cycle detected (graph is not a DAG)");

	compiled_graph graph;
	graph.steps.reserve(order.size());

	for (std::size_t idx : order) {
		graph.steps.emplace_back(compiled_graph::step {
			.buffer_memory_barriers	= std::move(tmp[idx].buffer_memory_barriers),
			.record					= std::move(tmp[idx].record)
		});
	}

	return graph;
}

inline void execute(
	const compiled_graph& graph,
	vk::CommandBuffer cmd
) {
	for (const auto& step : graph.steps) {
		//for (const auto& dep : step.dependencies)
		//	cmd.pipelineBarrier2(dep);
		//step.record(cmd);
	}
}

}