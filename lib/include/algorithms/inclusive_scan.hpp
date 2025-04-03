#pragma once

#include <allocator.hpp>
#include <algorithms/dispatch.hpp>
#include <detailed_exception.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

void inclusive_scan(
	typed_buffer<uint16_t>& input,
	typed_buffer<uint16_t>& output,
	typed_buffer<uint16_t>& group_sums,
	shader_manager& shader_manager,
	vk::CommandBuffer cmd_buffer
) {
	if (input.size() != output.size())
		throw detailed_exception("Input and output buffers must be the same size");

	std::array<uint32_t, 3> dispatch_counts = { 128, 1, 1 };
	uint32_t group_count = (input.size() + dispatch_counts[0] - 1) / dispatch_counts[0];

	if (group_sums.size() < group_count)
		throw detailed_exception("Group sums buffer is too small");

	auto inclusive_scan_shader = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) + "\\inclusive_scan.slang",
		"workgroup_inclusive_scan",
		{ 128, 1, 1 }
	);

	auto subgroup_exclusive_scan_shader = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) = "\\inclusive_scan.slang",
		"subgroup_exclusive_scan",
		{ 1, 1, 1 }
	);

	auto propogate_group_sums_shader = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) = "\\inclusive_scan.slang",
		"propogate_group_sums",
		{ group_count, 1, 1 }
	);

	struct inclusive_span_push_constants {
		device_span<uint16_t> input;
		device_span<uint16_t> output;
		device_span<uint16_t> group_sums;
	};

	inclusive_span_push_constants push_constants = {
		.input = input.device_span(),
		.output = output.device_span(),
		.group_sums = group_sums.device_span()
	};

	dispatch_shader<inclusive_span_push_constants>(
		cmd_buffer,
		inclusive_scan_shader,
		dispatch_counts,
		vk::ShaderStageFlagBits::eCompute,
		push_constants
	);

	cmd_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(barrier));
}

} // namespace vkengine