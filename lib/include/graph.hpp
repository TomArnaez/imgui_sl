#pragma once

#include <detailed_exception.hpp>
#include <unordered_map>
#include <functional>
#include <variant>
#include <queue>

namespace vkengine {

struct queue_info {
	vk::Queue			queue;
	uint32_t			family;
	vk::QueueFlagBits	capabilities;
};

struct buffer_usage {
	vk::Buffer				buffer;
	vk::AccessFlags2		access;
	vk::PipelineStageFlags2 stages;

	bool is_write() const noexcept {
		constexpr auto w =
			vk::AccessFlagBits2::eMemoryWrite |
			vk::AccessFlagBits2::eShaderWrite |
			vk::AccessFlagBits2::eTransferWrite |
			vk::AccessFlagBits2::eHostWrite;
		return static_cast<bool>(access & w);
	}
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

template<operation... ops>
using op_variant = std::variant<ops...>;

class graph_builder {
public:
	graph_builder() = default;
	graph_builder(const graph_builder&) = delete;
	graph_builder& operator=(const graph_builder&) = delete;

	enum class graph_err {
		cycle_detected,
		duplicate_resource,
		unsupported_usage,
		internal_bug,
	};

	template<operation... Ops>
	[[nodiscard]]
	auto build(const std::vector<op_variant<Ops...>>& ops)
		-> std::expected<compiled_graph, graph_err> {
		nodes_.reserve(ops.size());
		for (auto& v : ops) nodes_.emplace_back();

		std::size_t idx = 0;
		for (auto& variant : ops) {
			std::visit([&]<typename Op>(const Op& op) {
				analyse_op(op, idx);
			}, variant);
			++idx;
		}

		wire_barriers();

		if (auto r = topo_order(); !r) return std::unexpected(r.error());
		return emit_graph();
	}

private:
	struct node {
		std::vector<buffer_usage>				usages;
		std::vector<vk::BufferMemoryBarrier2>	barriers;
		std::vector<vk::BufferMemoryBarrier2>   pre_barriers;
		std::vector<vk::BufferMemoryBarrier2>   post_barriers;
		std::function<void(vk::CommandBuffer)>	record;
		std::vector<std::size_t>				out_edges;
		uint32_t								queue_family;
	};

	struct last_use {
		buffer_usage	usage;
		size_t			node_idx;
		uint32_t		queue_family;
	};

	template<typename Op>
	void analyse_op(const Op& op, std::size_t node_idx) {
		auto& n = nodes_[node_idx];
		n.record = [op](vk::CommandBuffer cb) { op.record(cb); };
		std::ranges::copy(op.usages(), std::back_inserter(n.usages));
	}

	void wire_barriers() {
		for (size_t i = 0; i < nodes_.size(); ++i) {
			auto& curr_node = nodes_[i];

			for (const buffer_usage& curr : curr_node.usages) {
				auto [it, first_time] = last_.try_emplace(
					curr.buffer, last_use{ curr, i });

				if (first_time) continue;

				auto& prev_use = it->second;

				auto& [prev_usage, prev_node_idx, prev_queue_family] = it->second;

				const bool need_order		= prev_usage.is_write() || curr.is_write();
				const bool queue_changed	= prev_queue_family != curr_node.queue_family;

				if (need_order && prev_node_idx != i) {

					if (queue_changed) {
						// (a) split into release (prev) + acquire (curr)
						prev_queue_family_release(prev_use, curr, curr_node.queue_family, i);
					}

					nodes_[i].barriers.emplace_back(
						vk::BufferMemoryBarrier2{}
						.setSrcStageMask(prev_usage.stages)
						.setDstStageMask(curr.stages)
						.setSrcAccessMask(prev_usage.access)
						.setDstAccessMask(curr.access)
						.setBuffer(curr.buffer)
						.setOffset(0)
						.setSize(VK_WHOLE_SIZE));

					nodes_[prev_node_idx].out_edges.push_back(i);
				}

				prev_usage.access	|= curr.access;
				prev_usage.stages	|= curr.stages;
				prev_node_idx		= i;
				prev_queue_family	= curr_node.queue_family;
			}
		}
	}

	//------------------------------------------------------------------
	//  helpers to push release / acquire barriers
	//------------------------------------------------------------------

	void prev_queue_family_release(const last_use& prev, const buffer_usage& curr, 
		uint32_t dst_family, size_t dst_node_idx) {
		auto& prev_node = nodes_[prev.node_idx];
		prev_node.post_barriers.emplace_back(
			vk::BufferMemoryBarrier2()
			.setSrcStageMask(prev.usage.stages)
			.setSrcAccessMask(prev.usage.access)
			.setDstStageMask(vk::PipelineStageFlagBits2::eNone)
			.setDstAccessMask(vk::AccessFlagBits2::eNone)
			.setSrcQueueFamilyIndex(prev.queue_family)
			.setDstQueueFamilyIndex(dst_family)
			.setBuffer(curr.buffer)
			.setOffset(0)
			.setSize(VK_WHOLE_SIZE));

		prev_node.out_edges.push_back(dst_node_idx);
	}

	void curr_family_acquire(const last_use& prev, const buffer_usage& curr,
		uint32_t curr_family, node& curr_node) {
		curr_node.pre_barriers.emplace_back(
			vk::BufferMemoryBarrier2{}
			.setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
			.setSrcAccessMask(vk::AccessFlagBits2::eNone)
			.setDstStageMask(curr.stages)
			.setDstAccessMask(curr.access)
			.setSrcQueueFamilyIndex(prev.queue_family)
			.setDstQueueFamilyIndex(curr_family)
			.setBuffer(curr.buffer)
			.setOffset(0)
			.setSize(VK_WHOLE_SIZE));
	}

	auto topo_order() -> std::expected<void, graph_err> {
		std::vector<size_t> indeg(nodes_.size(), 0);

		for (const auto& n : nodes_)
			for (auto v : n.out_edges) ++indeg[v];

		std::vector<size_t> queue(std::from_range, std::views::iota(0uz, indeg.size()) | std::views::filter([&](auto i) { return indeg[i] == 0; }));
		order_.reserve(nodes_.size());

		while (!queue.empty()) {
			auto u = queue.back();
			queue.pop_back();
			order_.push_back(u);
			for (auto v : nodes_[u].out_edges)
				if (--indeg[v] == 0) queue.push_back(v);
		}

		if (order_.size() != nodes_.size())
			return std::unexpected(graph_err::cycle_detected);

		return {};
	}

	auto emit_graph() -> compiled_graph
	{
		compiled_graph g;
		g.steps.reserve(nodes_.size());

		for (auto idx : order_)
			g.steps.emplace_back(compiled_graph::step {
				.buffer_memory_barriers = std::move(nodes_[idx].barriers),
				.record = std::move(nodes_[idx].record)
			});

		return g;
	}

	std::unordered_map<
		vk::Buffer, last_use,
		std::hash<VkBuffer>,
		std::equal_to<>>		last_;
	std::vector<node>			nodes_;
	std::vector<std::size_t>	order_;
};

}