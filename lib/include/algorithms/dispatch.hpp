#pragma once

#include <vulkan/vulkan_handles.hpp>
#include <vector>
#include <cstring>

namespace vkengine {

constexpr uint32_t VULKAN_PUSH_CONSTANT_SIZE_LIMIT = 128;

void dispatch_shader_impl(
    vk::CommandBuffer cmd,
    const shader_entry_point& shader,
    const std::array<uint32_t, 3>& group_counts,
    vk::ShaderStageFlagBits shader_stage_flags
) {
    cmd.bindShadersEXT(shader_stage_flags, { shader.shader_ext });
    cmd.dispatch(group_counts[0], group_counts[1], group_counts[2]);
}

template<typename TPushConstants>
void dispatch_shader(
    vk::CommandBuffer cmd,
    const shader_entry_point& shader,
    const std::array<uint32_t, 3>& group_counts,
    vk::ShaderStageFlagBits shader_stage_flags,
    const TPushConstants& push_constants
) {
    static_assert(std::is_trivially_copyable_v<TPushConstants>,
        "Push constants must be trivially copyable.");
    static_assert(sizeof(TPushConstants) <= VULKAN_PUSH_CONSTANT_SIZE_LIMIT,
        "Push constants size exceeds Vulkan limit.");
    assert(sizeof(TPushConstants) == shader.push_constant_range.size,
        "Push constants size mismatch shader's range.");

    cmd.pushConstants(
        shader.pipeline_layout,
        shader.push_constant_range.stageFlags,
        shader.push_constant_range.offset,
        sizeof(TPushConstants),
        &push_constants
    );

    dispatch_shader_impl(cmd, shader, group_counts, shader_stage_flags);
}

void dispatch_shader(
    vk::CommandBuffer cmd,
    const shader_entry_point& shader,
    const std::array<uint32_t, 3>& group_counts,
    vk::ShaderStageFlagBits shader_stage_flags
) {
    dispatch_shader_impl(cmd, shader, group_counts, shader_stage_flags);
}

} // namespace vkengine