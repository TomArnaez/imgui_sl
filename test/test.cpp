#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <allocator.hpp>
#include <algorithms/inclusive_scan.hpp>
#include <algorithms/histogram.hpp>
#include <algorithms/normalise.hpp>
#include <algorithms/median_filter.hpp>
#include <typed_buffer.hpp>
#include <graph.hpp>
#include <gpu.hpp>
#include <ranges>
#include <iostream>

#include <vulkan/vulkan_to_string.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

bool is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
    for (const auto& prop : properties)
        if (strcmp(prop.extensionName, extension) == 0)
            return true;
    return false;
}

}

struct vk_state {
    vk_state(vk::Instance instance, vkengine::gpu gpu, std::vector<const char*> device_extensions)
        : core(instance, gpu, device_extensions), allocator(core), shader_manager(core) { }

    vkengine::vulkan_core core;
    vkengine::allocator allocator;
    vkengine::shader_manager shader_manager;
};

int main() {
    spdlog::set_level(spdlog::level::debug);
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    std::vector<const char*> instance_extensions;
    vk::ApplicationInfo app_info("Dear ImGui Vulkan App", 1, "No Engine", 1, VK_API_VERSION_1_4);

    auto available_extensions = vk::enumerateInstanceExtensionProperties(nullptr);
    if (is_extension_available(available_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    instance_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

    std::vector<const char*> layers;
#ifdef APP_USE_VULKAN_DEBUG_UTILS
    layers.emplace_back("VK_LAYER_KHRONOS_validation");
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    auto validation_features_enables = std::array{ vk::ValidationFeatureEnableEXT::eDebugPrintf };
    auto validation_features = vk::ValidationFeaturesEXT()
        .setEnabledValidationFeatures(validation_features_enables);

    auto instance_ci = vk::InstanceCreateInfo()
        .setPApplicationInfo(&app_info)
        .setPEnabledExtensionNames(instance_extensions)
        .setPEnabledLayerNames(layers)
        .setPNext(&validation_features);

    vk::Instance instance = vk::createInstance(instance_ci);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    std::vector<const char*> device_extensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    };

    auto gpus = vkengine::enumerate_gpus(instance);

    for (auto& gpu : gpus) {
		std::cout << "GPU name" << gpu.properties.properties.deviceName << std::endl;
		std::cout << "Subgroup size: " << gpu.subgroup_properties.subgroupSize << std::endl;
    }

    vk_state state(instance, gpus[0], device_extensions);
}