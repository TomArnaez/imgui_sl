#pragma once

#include <algorithms/dispatch.hpp>
#include <algorithms/types.hpp>

namespace vkengine {

constexpr uint32_t NORMALISE_WORKGROUP_SIZE_X = 128;

template<typename T, typename U>
struct normalise_push_constants {
	device_span input;
	device_span output;
	T input_min;
	T input_max;
	U min;
	U max;
};

template<typename T, typename U, access_policy policy>
void normalise(
	typed_buffer<T, policy>& input,
	typed_buffer<U, policy>& output,
	T input_min,
	T input_max,
	U min,
	U max,
	shader_manager& shader_manager,
	vk::CommandBuffer cmd_buffer
) {
	if (input.size() != output.size())
		throw detailed_exception("Input and output buffers must be the same size");

	auto workgroup_module = shader_manager.create_shader_module_from_source_string(
		fmt::format(
			"export static const uint NORMALISE_WORKGROUP_SIZE_X = {};",
			NORMALISE_WORKGROUP_SIZE_X
		),
		"workgroup_module"
	);

	shader_manager::entry_point_compile_info entry_point_compile_info = {
		.name = "normalise",
		//.specialisation_type_names = {
		//	type_name<T>::value,
		//	type_name<U>::value
		//}
	};

	auto shader_object = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) + "/normalise.slang",
		{ entry_point_compile_info },
		{ workgroup_module }
	);

	normalise_push_constants<T, U> push_constants = {
		.input = input,
		.output = output,
		.input_min = input_min,
		.input_max = input_max,
		.min = min,
		.max = max
	};

	std::array<uint32_t, 3> workgroup_counts = { (input.size() + NORMALISE_WORKGROUP_SIZE_X - 1) / NORMALISE_WORKGROUP_SIZE_X, 1, 1 };

	dispatch_shader(
		cmd_buffer,
		shader_object[0],
		workgroup_counts,
		vk::ShaderStageFlagBits::eCompute,
		push_constants
	);
}

}