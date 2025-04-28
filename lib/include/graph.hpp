#pragma once

#include <unordered_map>
#include <variant>
#include <queue>

namespace vkengine {

struct buffer_usage {
	vk::Buffer				buffer;
	vk::AccessFlags2		access;
	vk::PipelineStageFlags2 stages;
	uint32_t				queue;
};

template<typename Op>
concept operation = requires(Op op, vk::CommandBuffer cb) {
	{ op.usages() }	-> std::ranges::forward_range;
	{ op.record(cb) }	-> std::same_as<void>;
};

struct test_op {
	std::array<buffer_usage, 1> buffer_usages;
	std::span<const buffer_usage> usages() const {
		return buffer_usages;
	}

	void record(vk::CommandBuffer cb) {
	}
};

struct compiled_graph {
	struct step {
		std::vector<vk::DependencyInfo>			dependencies;
		std::function<void(vk::CommandBuffer)>	record;
		uint32_t queue_index;
	};

	std::vector<step> steps;
};

constexpr bool is_write(vk::AccessFlags2 a) noexcept {
	return static_cast<bool>(a & vk::AccessFlagBits2::eMemoryWrite);
}

template<operation... ops>
using op_variant = std::variant<ops...>;

template<operation... ops>
compiled_graph compile(const std::vector<op_variant<ops...>>& nodes) {
	using variant = std::variant<ops...>;
	using last_key = std::tuple<vk::Buffer, uint32_t>;			// buffer + queue

	struct last_use {
		vk::AccessFlags2        access{};
		vk::PipelineStageFlags2 stages{};
		uint32_t                queue_index{};
		std::size_t             node_index{};   // who produced it
	};

	struct tmp_node {
		std::vector<vk::DependencyInfo>			deps;			// barriers attached here
		uint32_t								queue_index{};
		std::function<void(vk::CommandBuffer)>	record;
		std::vector<std::size_t>				out_edges;		// DAG edges for topo sort
	};

	std::vector<tmp_node> tmp(nodes.size());
	std::unordered_map<last_key, last_use> last;  // last reader/writer table

	for (std::size_t i = 0; i < nodes.size(); ++i) {
		const variant& node = nodes[i];

		std::visit([&, i](const auto& op) { 
			tmp[i].record = [op](vk::CommandBuffer cb) { op.record(cb); };

			for (const buffer_usage& usage : op.usages()) {
				last_key key{ usage.buffer, usage.queue };

				if (auto it = last.find(key); it != last.end()) {
					last_use& prev = it->second;
					bool need_order = is_write(prev.access) || is_write(u.access);
					if (need_order) {
						auto barrier = vk::BufferMemoryBarrier2()
							.setSrcStageMask(prev.stages)
							.setDstStageMask(usage.stages)
							.setSrcAccessMask(prev.access)
							.setDstAccessMask(usage.access)
					}
				})
			}
		}, node);
	}
}

}