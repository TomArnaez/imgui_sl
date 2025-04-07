#pragma once

#include <algorithms/dispatch.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

void calculate_histogram(
	typed_buffer<uint16_t>& input,
	typed_buffer<uint32_t>& histogram,
	shader_manager& shader_manager,
	vk::CommandBuffer cmd_buffer
) {
	constexpr std::array<uint32_t, 3>	workgroup_sizes = { 512, 1, 1 };
	std::array<uint32_t, 3>				dispatch_counts = { (input.size() + workgroup_sizes[0] - 1) / workgroup_sizes[0], 1, 1 };

	//auto histogram_shader = shader_manager.load_shader(
	//	std::string(VKENGINE_SHADER_DIR) + "/histogram.slang",
	//	{ "caculate_histogram" },
	//	dispatch_counts
	//)[0];

	//struct histogram_push_constants {
	//	device_span<uint16_t> input;
	//	device_span<uint32_t> histogram;
	//}; 

	//histogram_push_constants histogram_push_constants = {
	//	.input = input.device_span(),
	//	.histogram = histogram.device_span()
	//};

	//dispatch_shader(
	//	cmd_buffer,
	//	histogram_shader,
	//	dispatch_counts,
	//	vk::ShaderStageFlagBits::eCompute,
	//	histogram_push_constants
	//);
}

} // namespace vkengine