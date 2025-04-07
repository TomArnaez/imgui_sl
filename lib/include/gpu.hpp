#pragma once

#include <vulkan/vulkan.hpp>
#include <ranges>
#include <vector>

namespace vkengine {

struct gpu {
    vk::PhysicalDevice                      physical_device;
    vk::PhysicalDeviceProperties2           properties;
    vk::PhysicalDeviceFeatures2             features;
    vk::PhysicalDeviceMemoryProperties2     memory_properties;
    std::vector<vk::QueueFamilyProperties>  queue_family_properties;
    vk::PhysicalDeviceSubgroupProperties    subgroup_properties;
};

[[nodiscard]] inline std::vector<gpu> enumerate_gpus(vk::Instance instance) {
    return instance.enumeratePhysicalDevices() | std::views::transform([](vk::PhysicalDevice phys_dev) {
        auto properties = phys_dev.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();

        gpu g;

        g.physical_device = phys_dev;
		g.properties = properties.get<vk::PhysicalDeviceProperties2>();
		g.subgroup_properties = properties.get<vk::PhysicalDeviceSubgroupProperties>();
        g.features = phys_dev.getFeatures2();
        g.memory_properties = phys_dev.getMemoryProperties2();
        g.queue_family_properties = phys_dev.getQueueFamilyProperties();

        return g;
        }) | std::ranges::to<std::vector>();
}
}