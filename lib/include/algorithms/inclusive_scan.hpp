#pragma once

#include <allocator.hpp>
#include <algorithms/dispatch.hpp>
#include <detailed_exception.hpp>
#include <shader_manager.hpp>
#include <typed_buffer.hpp>

namespace vkengine {

template<access_policy policy>
void inclusive_scan(
	typed_buffer<uint32_t, policy>& input,
	typed_buffer<uint32_t, policy>& output,
	typed_buffer<uint32_t, policy>& group_sums,
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
		std::string(VKENGINE_SHADER_DIR) + "/inclusive_scan.slang",
		{ .name = "workgroup_inclusive_scan" },
		dispatch_counts
	)[0];

	auto subgroup_exclusive_scan_shader = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) + "/inclusive_scan.slang",
		{ .name = "subgroup_exclusive_scan" },
		{ 1, 1, 1 }
	)[0];

	auto propogate_group_sums_shader = shader_manager.load_shader(
		std::string(VKENGINE_SHADER_DIR) + "/inclusive_scan.slang",
		{ .name = "propogate_group_sums" },
		{ group_count, 1, 1 }
	)[0];

	device_span<uint32_t> input_span = input.device_span();
	device_span<uint32_t> output_span = output.device_span();
	device_span<uint32_t> group_sums_span = group_sums.device_span();

	struct inclusive_span_push_constants {
		device_span<uint32_t> input;
		device_span<uint32_t> output;
		device_span<uint32_t> group_sums;
	};

	inclusive_span_push_constants scan_push_constants = {
		.input = input_span,
		.output = output_span,
		.group_sums = group_sums_span
	};

	dispatch_shader(
		cmd_buffer,
		inclusive_scan_shader,
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

	dispatch_shader<device_span<uint32_t>>(
		cmd_buffer,
		subgroup_exclusive_scan_shader,
		{1, 1, 1},
		vk::ShaderStageFlagBits::eCompute,
		group_sums_span
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
		propogate_group_sums_shader,
		{group_count, 1, 1},
		vk::ShaderStageFlagBits::eCompute,
		scan_push_constants
	);
}

} // namespace vkengine