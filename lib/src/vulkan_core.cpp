#include <vulkan/vulkan.hpp>
#include <vulkan_core.hpp>
#include <spdlog/spdlog.h>
#include <detailed_exception.hpp>

namespace vkengine {

bool vulkan_core::is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
    for (const auto& prop : properties)
        if (strcmp(prop.extensionName, extension) == 0)
            return true;
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkan_core::debug_report_callback(
    VkDebugReportFlagsEXT      flags,
    VkDebugReportObjectTypeEXT object_type,
    uint64_t                   object,
    size_t                     location,
    int32_t                    message_code,
    const char* p_layer_prefix,
    const char* p_message,
    void* p_user_data
) {
    spdlog::error("Validation Layer ({}): {}", p_layer_prefix, p_message);
    return VK_FALSE;
}

uint32_t vulkan_core::find_dedicated_transfer_family(const std::vector<vk::QueueFamilyProperties>& families) {
    // Look for transfer-only family
    for (uint32_t i = 0; i < families.size(); ++i) {
        const auto& props = families[i];
        if ((props.queueFlags & vk::QueueFlagBits::eTransfer) &&
            !(props.queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
            return i;
        }
    }
    // Fallback to any transfer-capable family
    for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eTransfer) {
            return i;
        }
    }
    throw detailed_exception("No transfer queue family found");
}

uint32_t vulkan_core::find_compute_family(const std::vector<vk::QueueFamilyProperties>& families) {
    // Look for compute-only family
    for (uint32_t i = 0; i < families.size(); ++i) {
        const auto& props = families[i];
        if ((props.queueFlags & vk::QueueFlagBits::eCompute) &&
            !(props.queueFlags & vk::QueueFlagBits::eGraphics)) {
            return i;
        }
    }
    // Fallback to graphics family (which always includes compute)
    for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return i;
    }
    throw detailed_exception("No compute queue family found");
}

uint32_t vulkan_core::find_graphics_family(const std::vector<vk::QueueFamilyProperties>& families) {
    for (uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return i;
    }
    throw detailed_exception("No graphics queue family found");
}

vulkan_core::vulkan_core(vk::Instance instance, const vkengine::gpu& gpu, std::vector<const char*> device_extensions)
    : instance_(instance), gpu_(gpu) {
#ifdef APP_USE_VULKAN_DEBUG_REPORT
    debug_report_ = instance_.createDebugReportCallbackEXT(
        vk::DebugReportCallbackCreateInfoEXT()
        .setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::ePerformanceWarning)
        .setPfnCallback(debug_report_callback)
    );
#endif

    transfer_queue_family_ = find_dedicated_transfer_family(gpu_.queue_family_properties);
    compute_queue_family_ = find_compute_family(gpu_.queue_family_properties);
    graphics_queue_family_ = find_graphics_family(gpu_.queue_family_properties);

    std::unordered_map<uint32_t, uint32_t> queue_counts;
    queue_counts[transfer_queue_family_]++;
    queue_counts[compute_queue_family_]++;
    queue_counts[graphics_queue_family_]++;

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::vector<std::vector<float>> priorities_storage;

    for (const auto& [family, count] : queue_counts) {
        std::vector<float> priorities(count, 1.0f);
        priorities_storage.emplace_back(priorities);
        queue_create_infos.emplace_back(
            vk::DeviceQueueCreateInfo()
            .setQueueFamilyIndex(family)
            .setQueueCount(count)
            .setPQueuePriorities(priorities_storage.back().data())
        );
    }

    auto device_create_info_chain = vk::StructureChain
        (
            vk::DeviceCreateInfo()
                .setQueueCreateInfos(queue_create_infos)
                .setPEnabledExtensionNames(device_extensions),
            vk::PhysicalDeviceTimelineSemaphoreFeatures()
                .setTimelineSemaphore(true),
            vk::PhysicalDeviceDynamicRenderingFeatures()
                .setDynamicRendering(true),
            vk::PhysicalDeviceShaderObjectFeaturesEXT()
                .setShaderObject(true),
            vk::PhysicalDeviceBufferDeviceAddressFeatures()
                .setBufferDeviceAddress(true),
            vk::PhysicalDeviceSynchronization2Features()
                .setSynchronization2(true)
        );

    device_ = gpu_.physical_device.createDevice(device_create_info_chain.get<vk::DeviceCreateInfo>());
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

    std::unordered_map<uint32_t, uint32_t> family_indices;
    transfer_queue_ = device_.getQueue(transfer_queue_family_, family_indices[transfer_queue_family_]++);
    compute_queue_ = device_.getQueue(compute_queue_family_, family_indices[compute_queue_family_]++);
    graphics_queue_ = device_.getQueue(graphics_queue_family_, family_indices[graphics_queue_family_]++);

    transfer_pool_ = device_.createCommandPool(
        vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(transfer_queue_family_)
        .setFlags(vk::CommandPoolCreateFlagBits::eTransient |  // Optimized for short-lived buffers
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow buffer resets
    );

    compute_pool_ = device_.createCommandPool(
        vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(compute_queue_family_)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow individual buffer resets
    );

    graphics_pool_ = device_.createCommandPool(
        vk::CommandPoolCreateInfo()
        .setQueueFamilyIndex(graphics_queue_family_)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)  // Allow individual buffer resets
    );
}

vulkan_core::~vulkan_core() {
    device_.destroyCommandPool(transfer_pool_);
    device_.destroyCommandPool(compute_pool_);
    device_.destroyCommandPool(graphics_pool_);
}

vk::Instance vulkan_core::instance() const {
    return instance_;
}

vk::Device vulkan_core::device() const {
    return device_;
}

vk::PhysicalDevice vulkan_core::physical_device() const {
	return gpu_.physical_device;
}

gpu vulkan_core::gpu() const {
	return gpu_;
}

uint32_t vulkan_core::graphics_queue_family() const {
    return graphics_queue_family_;
}

vk::CommandPool vulkan_core::graphics_command_pool() const {
    return graphics_pool_;
}

vk::CommandPool vulkan_core::transfer_command_pool() const {
    return transfer_pool_;
}

vk::CommandPool vulkan_core::compute_command_pool() const {
	return compute_pool_;
}

vk::Queue vulkan_core::graphics_queue() const {
    return graphics_queue_;
}

vk::Queue vulkan_core::compute_queue() const {
    return compute_queue_;
}

vk::Queue vulkan_core::transfer_queue() const {
    return transfer_queue_;
}

}