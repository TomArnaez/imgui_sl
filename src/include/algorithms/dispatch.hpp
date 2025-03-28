#pragma once

#include <vulkan/vulkan_handles.hpp>
#include <vector>
#include <cstring>

namespace vkengine {

void launch_compute_shader_impl(
    vk::CommandBuffer cmd,
    const shader_object& shader,
    std::array<uint32_t, 3> group_counts
) {
    cmd.bindShadersEXT(vk::ShaderStageFlagBits::eCompute, { shader.shader_ext });
    cmd.dispatch(group_counts[0], group_counts[1], group_counts[2]);
}

template<typename TPushConstants>
void launch_compute_shader(
    vk::CommandBuffer cmd,
    const shader_object& shader,
    std::array<uint32_t, 3> group_counts,
    const TPushConstants* push_constants = nullptr
) {
    static_assert(std::is_trivially_copyable_v<TPushConstants>,
        "Push constants must be trivially copyable.");
    static_assert(sizeof(TPushConstants) <= 128,
        "Push constants size exceeds Vulkan limit.");

    if (push_constants) {
        assert(sizeof(TPushConstants) == shader.push_constant_range.size &&
            "Push constants size mismatch shader's range.");

        cmd.pushConstants(
            shader.pipeline_layout,
            shader.push_constant_range.stageFlags,
            shader.push_constant_range.offset,
            sizeof(TPushConstants),
            push_constants
        );
    }

    launch_compute_shader_impl(cmd, shader, group_counts);
}

template<typename... Args>
void launch_compute_shader(
    vk::CommandBuffer cmd,
    const shader_object& shader,
    std::array<uint32_t, 3> group_counts,
    Args&&... args
) {
    static_assert((std::is_trivially_copyable_v<Args> && ...),
        "All arguments must be trivially copyable.");

    const size_t total_size = (sizeof(Args) + ...);
    assert(total_size == shader.push_constant_range.size &&
        "Arguments size mismatch shader's push constant size.");

    if constexpr (sizeof...(Args) > 0) {
        std::vector<std::byte> buffer(total_size);
        size_t offset = 0;

        // Corrected fold expression syntax
        ((std::memcpy(buffer.data() + offset, &args, sizeof(args)),
            offset += sizeof(args)), ...);

        cmd.pushConstants(
            shader.pipeline_layout,
            shader.push_constant_range.stageFlags,
            shader.push_constant_range.offset,
            total_size,
            buffer.data()
        );
    }

    launch_compute_shader_impl(cmd, shader, group_counts);
}

} // namespace vkengine