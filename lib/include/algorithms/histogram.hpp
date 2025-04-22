#pragma once

#include <algorithms/dispatch.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

constexpr uint32_t HISTOGRAM_WORKGROUP_SIZE_X = 128;

struct histogram_push_constants {
	device_span input;
	device_span histogram;
};

class histogram_operator {
public:
	histogram_operator(shader_manager& shader_manager) {
		auto workgroup_module = shader_manager.create_shader_module_from_source_string(
			fmt::format(
				"export static const uint HISTOGRAM_WORKGROUP_SIZE_X = {};",
				HISTOGRAM_WORKGROUP_SIZE_X
			), "workgroup_module");

		histogram_shader_program_ = shader_manager.load_shader(
			std::string(VKENGINE_SHADER_DIR) + "/histogram.slang",
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
		std::array<uint32_t, 3>	dispatch_counts = { (input.size() + workgroup_sizes[0] - 1) / workgroup_sizes[0], 1, 1 };

		histogram_push_constants histogram_push_constants = {
			.input = input.device_span(),
			.histogram = output_histogram.device_span()
		};

		dispatch_shader(
			cmd_buffer,
			histogram_shader_program_.entry_points[0],
			dispatch_counts,
			vk::ShaderStageFlagBits::eCompute,
			histogram_push_constants
		);
	}
private:
	static constexpr std::array<uint32_t, 3>	workgroup_sizes = { 512, 1, 1 };
	shader_program								histogram_shader_program_;
};

} // namespace vkengine