#pragma once

#include <vulkan/vulkan.hpp>
#include <utility/slot_map.hpp>

#include <expected>
#include <functional>
#include <string_view>

struct resource_state {
	vk::PipelineStageFlags2 stage				= {};
	vk::AccessFlags2		access				= {};
	uint32_t				queue_family		= vk::QueueFamilyIgnored;
};

struct buffer_state {
	resource_state	resource_state;
	vk::Buffer		buffer;
};

struct image_state {
	resource_state	resource_state;
	vk::ImageLayout image_layout	= vk::ImageLayout::eUndefined;
	vk::Image		image	
};

class resource_manager {
public:
	resource_manager(vk::Device device)
		: device_(device) { }

	auto buffer(slot_map<buffer_state>::id id) const {
		return buffer_state_[id];
	}

	auto image(slot_map<image_state>::id id) const {
		return image_state_[id];
	}

	auto add_buffer(vk::Buffer buffer) {
		return buffer_state_.emplace(buffer);
	}

	auto add_image(vk::Image image) {
		return image_state_.emplace(arg);
	}
private:
	vk::Device				device_;

	slot_map<buffer_state>	buffer_state_;
	slot_map<image_state>	image_state_;
};

struct task_node {
};

struct task_graph {
};

struct compiled_pass {
	std::string_view							name;
	uint32_t									queue_idx;

	std::vector<resource_state>					acquire_barriers;
	std::vector<resource_state>					release_barriers;
	std::vector<vk::BufferMemoryBarrier2>		buffer_barriers;

	vk::CommandBuffer							cmd;
	std::move_only_function<vk::CommandBuffer>	body;

	std::vector<vk::SemaphoreSubmitInfo>		waits, signals;
};

class graph {
	std::vector<compiled_pass> compiled_pass;
};

template<class Body, class... Uses>
struct compute_pass {
	using uses_t = std::tuple<Uses...>;

	std::string_view name;
	body             body;
};

template<class BodyT = std::move_only_function<void(vk::CommandBuffer)>>
using any_pass = std::variant<
	compute_pass<BodyT>
>;

class graph_builder {
public:
	std::expected<graph, std::string> compile(
		vk::Device		device,
		vk::CommandPool pool,
		vk::Semaphore	semaphore
	) {
		if (passes_.empty()) return "no passes";

		std::vector<compiled_pass> out(passes_.size());
	}
private:
	void process_use() {

	}
	std::vector<any_pass> passes_;
};