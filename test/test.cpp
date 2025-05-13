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
#include <graph2.hpp>

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


struct writer_op {
    vk::Buffer                              buf;
    vkengine::buffer_usage                  use;

    explicit writer_op(vk::Buffer b)
        : buf{ b },
        use{ vkengine::buffer_usage{
                  b,
                  vk::AccessFlagBits2::eShaderWrite,
                  vk::PipelineStageFlagBits2::eComputeShader } } {
    }

    auto usages() const { return std::views::single(use); }
    void record(vk::CommandBuffer cb) const {
    }
};

struct reader_op {
    vk::Buffer                              buf;
    vkengine::buffer_usage                  use;

    explicit reader_op(vk::Buffer b)
        : buf{ b },
        use{ vkengine::buffer_usage{
                  b,
                  vk::AccessFlagBits2::eShaderRead,
                  vk::PipelineStageFlagBits2::eComputeShader } } {
    }

    auto usages() const { return std::views::single(use); }

    void record(vk::CommandBuffer cb) const {
    }
};

void test_graph(vk_state& state) {
    using test_node = vkengine::op_variant<writer_op, reader_op>;

    vkengine::host_visible_buffer<uint32_t> buffer_a(
        state.allocator,
        state.core,
        1024
    );

    std::vector<test_node> nodes;
    nodes.emplace_back(reader_op{ buffer_a.vk_handle() });
    nodes.emplace_back(writer_op{ buffer_a.vk_handle() });

    vkengine::compiled_graph graph = vkengine::graph_builder{}.build(nodes).value();

    for (auto& step : graph.steps) {
        auto& barriers = step.buffer_memory_barriers;

        for (auto& barrier : barriers) {
            std::cout << "srcAccessMask:\t" << vk::to_string(barrier.srcAccessMask) << std::endl;
            std::cout << "dstAccessMask:\t" << vk::to_string(barrier.dstAccessMask) << std::endl;
            std::cout << "srcStageMask:\t"  << vk::to_string(barrier.srcStageMask) << std::endl;
            std::cout << "dstStageMask:\t"  << vk::to_string(barrier.dstStageMask) << std::endl;
        }
    }
}

void test_normalisation(vk_state& state) {
    uint32_t element_count = 1024;

    vkengine::host_visible_buffer<uint32_t> input(
        state.allocator,
        state.core,
        element_count
    );

    vkengine::host_visible_buffer<uint16_t> output(
        state.allocator,
        state.core,
        element_count
    );

    auto&& range = input.data();
    std::ranges::copy(std::ranges::iota_view(0u, range.size()), range.begin());

    vk::CommandBuffer cmd_buffer = state.core.device().allocateCommandBuffers(
        vk::CommandBufferAllocateInfo()
        .setCommandPool(state.core.compute_command_pool())
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
        state.shader_manager,
        cmd_buffer
    );

    cmd_buffer.end();

    state.core.compute_queue().submit(
        vk::SubmitInfo()
        .setCommandBuffers(cmd_buffer)
    );

    state.core.device().waitIdle();

    for (uint32_t data : output.data())
        std::cout << data << " ";

    input.destroy();
    output.destroy();
}

void test_inclusive_scan(vk_state& state) {
    uint32_t element_count = 1024;

    vkengine::host_visible_buffer<uint32_t> buffer(
   	    state.allocator,
        state.core,
   	    element_count
   );

   vkengine::host_visible_buffer <uint32_t> group_sums(
       state.allocator,
       state.core,
       element_count
   );

   vkengine::host_visible_buffer<uint32_t> output(
       state.allocator,
       state.core,
       element_count
   );

    auto&& range = buffer.data();
   std::ranges::iota_view source(0u, range.size());
   std::ranges::copy(source, range.begin());

   vk::CommandBuffer cmd_buffer = state.core.device().allocateCommandBuffers(
   	vk::CommandBufferAllocateInfo()
   	.setCommandPool(state.core.compute_command_pool())
   	.setLevel(vk::CommandBufferLevel::ePrimary)
   	.setCommandBufferCount(1)
   ).front();

   cmd_buffer.begin(vk::CommandBufferBeginInfo()
   	.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
   );

   inclusive_scan(
   	buffer,
   	output,
   	group_sums,
   	state.shader_manager,
   	cmd_buffer
   );

   cmd_buffer.end();

   state.core.compute_queue().submit(
       vk::SubmitInfo()
   	    .setCommandBuffers(cmd_buffer)
   );

   state.core.device().waitIdle();

   for (uint32_t data : output.data())
       std::cout << data << " ";

   buffer.destroy();
   group_sums.destroy();
   output.destroy();
}

void test_median_filter(vk_state& state) {
    std::array<uint32_t, 2> shape = { 64, 64 };

    vkengine::host_visible_buffer_nd<uint16_t, 2> input(
        state.allocator,
        state.core,
        shape
    );

    vkengine::host_visible_buffer_nd<uint16_t, 2> output(
        state.allocator,
        state.core,
        shape
    );

    auto&& range = input.data();
    std::ranges::iota_view source(0u, shape[0] * shape[1]);
    std::ranges::copy(source, range.begin());

    vkengine::median_filter_operator median_filter_op(state.shader_manager);

    vk::CommandBuffer cmd_buffer = state.core.device().allocateCommandBuffers(
        vk::CommandBufferAllocateInfo()
        .setCommandPool(state.core.compute_command_pool())
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(1)
    ).front();

    cmd_buffer.begin(vk::CommandBufferBeginInfo()
        .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
    );

    median_filter_op.record(input, output, cmd_buffer);

    cmd_buffer.end();

    state.core.compute_queue().submit(
        vk::SubmitInfo()
        .setCommandBuffers(cmd_buffer)
    );

    state.core.device().waitIdle();

    for (uint16_t data : output.data())
        std::cout << data << std::endl;
}

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
    //test_median_filter(state);
    //test_inclusive_scan(state);
	//test_normalisation(state);
    test_graph(state);
}