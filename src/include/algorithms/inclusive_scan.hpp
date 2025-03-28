#pragma once

#include <allocator.hpp>
#include <algorithms/dispatch.hpp>
#include <detailed_exception.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

template<typename T, typename TAccumulation>
void inclusive_scan(
	typed_buffer<T>& input,
	typed_buffer<T>& output,
	typed_buffer<TAccumulation>& group_sums,
	const shader_manager& manager,
	vk::CommandBuffer cmd_buffer
) {
	if (input.size() != output.size())
		throw detailed_exception("Input and output buffers must be the same size");

	std::array<uint32_t, 3> workgroup_sizes = { 512, 1, 1 };
	uint32_t group_count = (input.size() + workgroup_sizes[0] - 1) / workgroup_sizes[0];

	if (group_sums.size() < group_count)
		throw detailed_exception("Group sums buffer is too small");

	device_span<T> input_span = input.device_span();
	device_span<T> output_span = output.device_span();
	device_span<TAccumulation> group_sums_span = group_sums.device_span();
}

} // namespace vkengine