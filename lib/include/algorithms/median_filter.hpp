#pragma once

#include <algorithms/dispatch.hpp>
#include <detailed_exception.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

constexpr uint32_t MEDIAN_FILTER_WORKGROUP_SIZE_X = 16;
constexpr uint32_t MEDIAN_FILTER_WORKGROUP_SIZE_Y = 16;

struct median_filter_push_constants {
	device_mdspan<2> input;
	device_mdspan<2> output;
};

class median_filter_operator {
	shader_object median_filter_shader_object;
public:
	median_filter_operator(shader_manager& shader_manager) {
		auto workgroup_module = shader_manager.create_shader_module_from_source_string(
			fmt::format(
				"export static const uint MEDIAN_FILTER_WORKGROUP_SIZE_X = {};"
				"export static const uint MEDIAN_FILTER_WORKGROUP_SIZE_Y = {};",
				MEDIAN_FILTER_WORKGROUP_SIZE_X, MEDIAN_FILTER_WORKGROUP_SIZE_Y
			), "workgroup_module" );

		median_filter_shader_object = shader_manager.load_shader(
			"median_filter",
			{ shader_manager::entry_point_compile_info { .name = "median_filter" } },
			{ workgroup_module }
		)[0];
	}

	template<access_policy policy>
	void record(
		typed_buffer<uint16_t, 2, policy>& input,
		typed_buffer<uint16_t, 2, policy>& output,
		vk::CommandBuffer cmd_buffer
	) {
		record_median_filter(input, output, median_filter_shader_object, cmd_buffer);
	}
};

template<access_policy policy>
void record_median_filter(
	typed_buffer<uint16_t, 2, policy>& input,
	typed_buffer<uint16_t, 2, policy>& output,
	shader_object& median_filter_shader_object,
	vk::CommandBuffer cmd_buffer
) {
	median_filter_push_constants push_constants = {
		.input = input,
		.output = output
	};

	std::array<uint32_t, 3> workgroup_counts = { 
		(input.shape()[1] + MEDIAN_FILTER_WORKGROUP_SIZE_X - 1) / MEDIAN_FILTER_WORKGROUP_SIZE_X,
		(input.shape()[0] + MEDIAN_FILTER_WORKGROUP_SIZE_Y - 1) / MEDIAN_FILTER_WORKGROUP_SIZE_Y,
		1 
	};

	dispatch_shader(
		cmd_buffer,
		median_filter_shader_object,
		workgroup_counts,
		vk::ShaderStageFlagBits::eCompute,
		push_constants
	);
}

}