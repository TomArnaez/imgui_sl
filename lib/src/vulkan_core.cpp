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

#ifdef APP_USE_VULKAN_DEBUG_UTILS
VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*                                       /*p_user_data*/) {

    const char* severity_str = "INFO";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity_str = "ERROR";
    }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity_str = "WARN";
    }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        severity_str = "TRACE";
    }

    spdlog::log(spdlog::level::from_str(severity_str),
        "Validation Layer ({}): {}",
        callback_data->pMessageIdName,
        callback_data->pMessage);

    return VK_FALSE;
}
#endif // APP_USE_VULKAN_DEBUG_UTILS

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
#ifdef APP_USE_VULKAN_DEBUG_UTILS
    debug_utils_messenger_ = instance_.createDebugUtilsMessengerEXT(
        vk::DebugUtilsMessengerCreateInfoEXT{}
        .setMessageSeverity(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
        .setMessageType(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance)
        .setPfnUserCallback(debug_utils_messenger_callback));
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
            vk::PhysicalDeviceFeatures2()
                .setFeatures(vk::PhysicalDeviceFeatures().setShaderInt16(true)),
            vk::PhysicalDeviceTimelineSemaphoreFeatures()
                .setTimelineSemaphore(true),
            vk::PhysicalDeviceDynamicRenderingFeatures()
                .setDynamicRendering(true),
            vk::PhysicalDeviceShaderObjectFeaturesEXT()
                .setShaderObject(true),
            vk::PhysicalDeviceBufferDeviceAddressFeatures()
                .setBufferDeviceAddress(true),
            vk::PhysicalDeviceSynchronization2Features()
                .setSynchronization2(true),
            vk::PhysicalDevice16BitStorageFeatures()
                .setStoragePushConstant16(true),
            vk::PhysicalDeviceScalarBlockLayoutFeatures()
                .setScalarBlockLayout(true)
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