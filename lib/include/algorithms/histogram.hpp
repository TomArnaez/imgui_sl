#pragma once

#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

template<typename T>
void calculate_histogram(
	typed_buffer<T>& input,
	typed_buffer<uint32_t>& histogram,
	const shader_manager& manager,
	vk::CommandBuffer cmd_buffer
) {
	std::array<uint32_t, 3> workgroup_sizes = { 512, 1, 1 };
	uint32_t group_count = (input.size() + workgroup_sizes[0] - 1) / workgroup_sizes[0];
}

} // namespace vkengine