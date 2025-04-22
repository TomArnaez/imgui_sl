#pragma once

#include <algorithms/dispatch.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

constexpr uint32_t HISTOGRAM_WORKGROUP_SIZE_X = 128;

class histogram_operator {
public:
	histogram_operator(shader_manager& shader_manager) {
		auto workgroup_module = shader_manager.create_shader_module_from_source_string(
			fmt::format(
				"export static const uint HISTOGRAM_WORKGROUP_SIZE_X = {};",
				HISTOGRAM_WORKGROUP_SIZE_X
			), "workgroup_module");

		histogram_shader_program_ = shader_manager.load_shader(
			"histogram",
			{ shader_manager::entry_point_compile_info {.name = "histogram" } },
			{ workgroup_module }
		);
	}

	template<access_policy policy>
	void record(
		vk::CommandBuffer cmd_buffer,
		typed_buffer<uint16_t, 2, policy>& input,
		typed_buffer<uint32_t, 1, policy>& output_histogram
	) {

	}
private:
	shader_program histogram_shader_program_;
};

template<access_policy policy>
void record_histogram(
	typed_buffer<uint16_t, 2, policy>& input,
	typed_buffer<uint32_t, 1, policy>& output_histogram
) {

}



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