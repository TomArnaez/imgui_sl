#pragma once

#include <algorithms/dispatch.hpp>
#include <detailed_exception.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

constexpr uint32_t INCLUSIVE_SCAN_WORKGROUP_SIZE = 128;

template<uint32_t dims, access_policy policy>
void inclusive_scan(
	typed_buffer<uint32_t, dims, policy>& input,
	typed_buffer<uint32_t, dims, policy>& output,
	typed_buffer<uint32_t, dims, policy>& group_sums,
	shader_manager& shader_manager,
	vk::CommandBuffer cmd_buffer
) {
	if (input.size() != output.size())
		throw detailed_exception("Input and output buffers must be the same size");

	std::array<uint32_t, 3> dispatch_counts = { 128, 1, 1 };
	uint32_t group_count = (input.size() + dispatch_counts[0] - 1) / dispatch_counts[0];

	if (group_sums.size() < group_count)
		throw detailed_exception("Group sums buffer is too small");

	auto workgroup_module = shader_manager.create_shader_module_from_source_string(
		fmt::format(
			"export static const uint INCLUSIVE_SCAN_WORKGROUP_SIZE = {};",
			INCLUSIVE_SCAN_WORKGROUP_SIZE
		),
		"workgroup_module"
	);

	auto shader_objects = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) + "/inclusive_scan.slang",
		{
			shader_manager::entry_point_compile_info {
				.name = "workgroup_inclusive_scan",
			},
			shader_manager::entry_point_compile_info {
				.name = "subgroup_exclusive_scan",
			},
			shader_manager::entry_point_compile_info {
				.name = "propogate_group_sums",
			}
		},
		{ workgroup_module }
	);

	struct inclusive_span_push_constants {
		device_span input;
		device_span output;
		device_span group_sums;
	};

	inclusive_span_push_constants scan_push_constants = {
		.input = input,
		.output = output,
		.group_sums = group_sums
	};

	dispatch_shader(
		cmd_buffer,
		shader_objects[0],
		dispatch_counts,
		vk::ShaderStageFlagBits::eCompute,
		scan_push_constants
	);

	auto inclusive_scan_barrier = vk::BufferMemoryBarrier2()
		.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
		.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
		.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
		.setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
		.setBuffer(group_sums.vk_handle())
		.setSize(VK_WHOLE_SIZE);


	cmd_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(inclusive_scan_barrier));

	dispatch_shader<device_span>(
		cmd_buffer,
		shader_objects[1],
		{1, 1, 1},
		vk::ShaderStageFlagBits::eCompute,
		group_sums
	);

	{
		std::array barriers = {
			vk::BufferMemoryBarrier2()
				.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setDstAccessMask(vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderWrite)
				.setBuffer(group_sums.vk_handle())
				.setSize(VK_WHOLE_SIZE),

			vk::BufferMemoryBarrier2()
				.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
				.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
				.setDstAccessMask(vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite)
				.setBuffer(output.vk_handle())
				.setSize(VK_WHOLE_SIZE)
		};

		cmd_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(barriers));
	}

	dispatch_shader(
		cmd_buffer,
		shader_objects[2],
		{group_count, 1, 1},
		vk::ShaderStageFlagBits::eCompute,
		scan_push_constants
	);
}

} // namespace vkengine