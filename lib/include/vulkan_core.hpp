#pragma once

#include <vulkan/vulkan_handles.hpp>
#include <gpu.hpp>

namespace vkengine {

class vulkan_core {
public:
    struct physical_device_props {
        vk::PhysicalDeviceProperties properties;
        vk::PhysicalDeviceSubgroupProperties subgroup_properties;
    };
private:
    vk::DebugUtilsMessengerEXT debug_utils_messenger_ = VK_NULL_HANDLE;

    vk::Instance        instance_;
	gpu                 gpu_;
    vk::Device          device_;
    vk::Queue           transfer_queue_;
    uint32_t            transfer_queue_family_;
    vk::Queue           compute_queue_;
    uint32_t            compute_queue_family_;
    vk::Queue           graphics_queue_;
    uint32_t            graphics_queue_family_;
    vk::CommandPool     transfer_pool_;
    vk::CommandPool     compute_pool_;
    vk::CommandPool     graphics_pool_;

    static bool is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report_callback(
        VkDebugReportFlagsEXT      flags,
        VkDebugReportObjectTypeEXT object_type,
        uint64_t                   object,
        size_t                     location,
        int32_t                    message_code,
        const char* p_layer_prefix,
        const char* p_message,
        void* p_user_data
    );

    uint32_t find_dedicated_transfer_family(const std::vector<vk::QueueFamilyProperties>& families);
    uint32_t find_compute_family(const std::vector<vk::QueueFamilyProperties>& families);
    uint32_t find_graphics_family(const std::vector<vk::QueueFamilyProperties>& families);
public:
    vulkan_core(vk::Instance instance, const gpu& gpu, std::vector<const char*> device_extensions);
    ~vulkan_core();

    vk::Instance instance() const;
    vk::Device device() const;
	vk::PhysicalDevice physical_device() const;
	gpu gpu() const;
    vk::Queue graphics_queue() const;
    uint32_t graphics_queue_family() const;
    vk::CommandPool graphics_command_pool() const;
    vk::CommandPool compute_command_pool() const;
    vk::CommandPool transfer_command_pool() const;
    vk::Queue compute_queue() const;
    vk::Queue transfer_queue() const;
};

}