#include <vulkan/vulkan.hpp>
#include <shader_manager.hpp>
#include <allocator.hpp>
#include <algorithms/inclusive_scan.hpp>
#include <algorithms/histogram.hpp>
#include <algorithms/normalise.hpp>
#include <queue_family.hpp>
#include <typed_buffer.hpp>
#include <gpu.hpp>
#include <ranges>
#include <iostream>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

bool is_extension_available(const std::vector<vk::ExtensionProperties>& properties, const char* extension) {
    for (const auto& prop : properties)
        if (strcmp(prop.extensionName, extension) == 0)
            return true;
    return false;
}

}

void test_normalisation(vkengine::vulkan_core& core, vkengine::allocator& allocator, vkengine::shader_manager& shader_manager) {
    uint32_t element_count = 1024;

    vkengine::host_visible_buffer<uint32_t> input(
        allocator,
        core,
        element_count
    );

    vkengine::host_visible_buffer<uint16_t> output(
        allocator,
        core,
        element_count
    );

    auto&& range = input.data();
    std::ranges::copy(std::ranges::iota_view(0u, range.size()), range.begin());

    vk::CommandBuffer cmd_buffer = core.device().allocateCommandBuffers(
        vk::CommandBufferAllocateInfo()
        .setCommandPool(core.compute_command_pool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1)
    ).front();

    cmd_buffer.begin(vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
    );

    normalise<uint32_t, uint16_t>(
        input,
        output,
        0,
        1024,
        0,
        255,
        shader_manager,
        cmd_buffer
    );

    cmd_buffer.end();

    core.compute_queue().submit(
        vk::SubmitInfo()
        .setCommandBuffers(cmd_buffer)
    );

    core.device().waitIdle();

    for (uint32_t data : output.data())
        std::cout << data << " ";
}

int main() {
    spdlog::set_level(spdlog::level::debug);
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    std::vector<const char*> instance_extensions;
    vk::ApplicationInfo app_info("Dear ImGui Vulkan App", 1, "No Engine", 1, VK_API_VERSION_1_4);
    auto available_extensions = vk::enumerateInstanceExtensionProperties(nullptr);
    if (is_extension_available(available_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    std::vector<const char*> layers;
#ifdef APP_USE_VULKAN_DEBUG_REPORT
    layers.emplace_back("VK_LAYER_KHRONOS_validation");
    instance_extensions.push_back("VK_EXT_debug_report");
#endif
    auto instance_ci = vk::InstanceCreateInfo()
        .setPApplicationInfo(&app_info)
        .setPEnabledExtensionNames(instance_extensions)
        .setPEnabledLayerNames(layers);

    vk::Instance instance = vk::createInstance(instance_ci);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    std::vector<const char*> device_extensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME
    };

    auto gpus = vkengine::enumerate_gpus(instance);

    for (auto& gpu : gpus) {
		std::cout << "GPU name" << gpu.properties.properties.deviceName << std::endl;
		std::cout << "Subgroup size: " << gpu.subgroup_properties.subgroupSize << std::endl;
    }

    vkengine::vulkan_core core(instance, gpus[0], device_extensions);
	vkengine::allocator allocator(core);
    vkengine::shader_manager shader_manager(core);

 //   VmaAllocationCreateInfo host_alloc_info = {};
 //   host_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
 //   host_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
 //       VMA_ALLOCATION_CREATE_MAPPED_BIT;

 //   uint32_t element_count = 1024;

 //   vkengine::host_visible_buffer<uint32_t> buffer(
	//	allocator,
	//	core,
	//	element_count
	//);

 //   vkengine::host_visible_buffer <uint32_t> group_sums(
 //       allocator,
 //       core,
 //       element_count
 //   );

 //   vkengine::host_visible_buffer<uint32_t> output(
 //       allocator,
 //       core,
 //       element_count
 //   );

 //   auto&& range = buffer.data();
 //   std::ranges::iota_view source(0u, range.size());
 //   std::ranges::copy(source, range.begin());

	//vk::CommandBuffer cmd_buffer = core.device().allocateCommandBuffers(
	//	vk::CommandBufferAllocateInfo()
	//	.setCommandPool(core.compute_command_pool())
	//	.setLevel(vk::CommandBufferLevel::ePrimary)
	//	.setCommandBufferCount(1)
	//).front();

	//cmd_buffer.begin(vk::CommandBufferBeginInfo()
	//	.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
	//);

	//inclusive_scan(
	//	buffer,
	//	output,
	//	group_sums,
	//	shader_manager,
	//	cmd_buffer
	//);

 //   cmd_buffer.end();

	//core.compute_queue().submit(
	//	vk::SubmitInfo()
	//	.setCommandBuffers(cmd_buffer)
	//);

	//core.device().waitIdle();

	//for (uint32_t data : output.data())
	//	std::cout << data << " ";

 //   buffer.destroy();
 //   group_sums.destroy();
	//output.destroy();

	test_normalisation(core, allocator, shader_manager);
}