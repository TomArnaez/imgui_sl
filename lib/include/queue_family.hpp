#pragma once

#include <vulkan/vulkan_structs.hpp>
#include <optional>

namespace vkengine {

inline std::optional<uint32_t> find_transfer_family(const std::vector<vk::QueueFamilyProperties>& families) {
    // Look for transfer-only family
    for (uint32_t i = 0; i < families.size(); ++i) {
        const auto& props = families[i];
        if ((props.queueFlags & vk::QueueFlagBits::eTransfer) &&
            !(props.queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
            return i;
        }
    }

    // Fallback to any transfer-capable family
    for (uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & vk::QueueFlagBits::eTransfer)
            return i;

    return std::nullopt;
}

inline std::optional<uint32_t> find_compute_family(const std::vector<vk::QueueFamilyProperties>& families) {
    // Look for compute-only family
    for (uint32_t i = 0; i < families.size(); ++i) {
        const auto& props = families[i];
        if ((props.queueFlags & vk::QueueFlagBits::eCompute) &&
            !(props.queueFlags & vk::QueueFlagBits::eGraphics))
            return i;
    }

    // Fallback to graphics family (which always includes compute)
    for (uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return i;

    return std::nullopt;
}

inline std::optional<uint32_t> find_graphics_family(const std::vector<vk::QueueFamilyProperties>& families) {
    for (uint32_t i = 0; i < families.size(); ++i)
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return i;

	return std::nullopt;
}

}