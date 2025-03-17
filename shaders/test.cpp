#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>
#include <string>
#include <cstdio>
#include <cstring>

#include <slang-com-ptr.h>
#include <slang.h>
#include <unordered_map>

#include "compute_dispatcher.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

template<typename... TArgs>
inline void report_error(const char* format, TArgs... args) {
    printf(format, std::forward<TArgs>(args)...);
}

inline void diagnose_if_needed(slang::IBlob* diagnostics_blob) {
    if (diagnostics_blob)
        report_error("%s", reinterpret_cast<const char*>(diagnostics_blob->getBufferPointer()));
}

struct VulkanContext {
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::Queue queue;
    uint32_t queueFamilyIndex;
};

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice physical_device) {
    auto memProperties = physical_device.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

VulkanContext initializeVulkan() {
    VulkanContext context;
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    vk::ApplicationInfo appInfo("VulkanCompute", 1, "No Engine", 1, VK_API_VERSION_1_4);
    
    vk::InstanceCreateInfo instanceCI({}, &appInfo);
    context.instance = vk::createInstanceUnique(instanceCI);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(context.instance.get());

    auto physicalDevices = context.instance->enumeratePhysicalDevices();
    for (const auto& pd : physicalDevices) {
        auto queueFamilies = pd.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                context.physical_device = pd;
                context.queueFamilyIndex = i;

                break;
            }
        }
        if (context.physical_device) break; // Break outer loop as soon as a device is found
    }

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCI({}, context.queueFamilyIndex, 1, &queuePriority);

    std::vector<const char*> deviceExtensions = {
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    };

    vk::PhysicalDeviceShaderObjectFeaturesEXT shader_object_features{};
    shader_object_features.shaderObject = VK_TRUE;

    vk::StructureChain<
        vk::DeviceCreateInfo,
        vk::PhysicalDeviceShaderObjectFeaturesEXT
    > device_chain;

    auto& device_create_info = device_chain.get<vk::DeviceCreateInfo>();
    device_create_info.setQueueCreateInfos(queueCI)
            .setPEnabledExtensionNames(deviceExtensions);

    device_chain.get<vk::PhysicalDeviceShaderObjectFeaturesEXT>() = shader_object_features;

    context.device = context.physical_device.createDevice(device_chain.get<vk::DeviceCreateInfo>());
    context.queue = context.device.getQueue(context.queueFamilyIndex, 0);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(context.device);

    return context;
}
constexpr uint32_t BUFFER_COUNT = 512;
constexpr uint32_t GROUP_SIZE_X = 128;
constexpr uint32_t NUM_GROUPS = (BUFFER_COUNT + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
constexpr uint32_t BUFFER_SIZE = BUFFER_COUNT * sizeof(uint32_t);

template<typename TPushConstants>
void launch_computer_shader(
    vk::CommandBuffer cmd,
    const shader_object& shader,
    std::array<uint32_t, 3> group_counts,
    const TPushConstants* push_constants_data = nullptr
) {
    cmd.bindShadersEXT(vk::ShaderStageFlagBits::eCompute, { shader.shader_ext });

    if (push_constants_data) {
        cmd.pushConstants(
            shader.pipeline_layout,
            shader.push_constant_range.stageFlags,
            shader.push_constant_range.offset,
            shader.push_constant_range.size,
            push_constants_data);
    }

    cmd.dispatch(group_counts[0], group_counts[1], group_counts[2]);
}

struct device_span {
    vk::DeviceAddress   data;
    uint32_t            size;
    uint32_t            _pad;
};

void launch_inclusive_scan(
    const VulkanContext& context,
    shader_object& inclusive_scan_shader,
    shader_object& exclusive_scan_object,
    shader_object& propogate_scan_shader
) {
    auto createBuffer = [&](vk::DeviceSize size) {
        return context.device.createBufferUnique({
            {},
            size,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::SharingMode::eExclusive
        });
    };

    auto inputBuffer = createBuffer(BUFFER_COUNT * sizeof(uint32_t));
    auto outputBuffer = createBuffer(BUFFER_COUNT * sizeof(uint32_t));
    auto groupSumsBuffer = createBuffer(NUM_GROUPS * sizeof(uint32_t));

    // Allocate and bind memory
    auto allocateMemory = [&](vk::Buffer buffer) {
        auto memReq = context.device.getBufferMemoryRequirements(buffer);
        auto memAlloc = vk::MemoryAllocateInfo(
            memReq.size,
            findMemoryType(memReq.memoryTypeBits, 
                           vk::MemoryPropertyFlagBits::eHostVisible | 
                           vk::MemoryPropertyFlagBits::eHostCoherent,
                           context.physical_device)
        );
        auto memory = context.device.allocateMemoryUnique(memAlloc);
        context.device.bindBufferMemory(buffer, memory.get(), 0);
        return memory;
    };

    auto inputMemory = allocateMemory(inputBuffer.get());
    auto outputMemory = allocateMemory(outputBuffer.get());
    auto groupSumsMemory = allocateMemory(groupSumsBuffer.get());

    // Initialize input data
    {
        auto* data = static_cast<uint32_t*>(context.device.mapMemory(inputMemory.get(), 0, BUFFER_SIZE));
        for (uint32_t i = 0; i < BUFFER_COUNT; ++i)
            data[i] = 1;
        context.device.unmapMemory(inputMemory.get());
    }

    auto getBufferAddress = [&](vk::Buffer buffer) {
        return context.device.getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer));
    };

    device_span input_span = {
        .data = getBufferAddress(inputBuffer.get()),
        .size = BUFFER_COUNT
    };

    device_span group_span = {
        .data = getBufferAddress(groupSumsBuffer.get()),
        .size = NUM_GROUPS
    };

    device_span output_span = {
        .data = getBufferAddress(outputBuffer.get()),
        .size = BUFFER_COUNT
    };

    #pragma pack(push, 1)
    struct inclusive_scan_push_constants {
        device_span input;
        device_span output;
        device_span groups;
    };
    #pragma pack(pop)

    #pragma pack(push, 1)
    struct exclusive_push_constants  {
        device_span groups;
    };
    #pragma pack(pop)

    static_assert(sizeof(inclusive_scan_push_constants) == 48, "PushConstants size mismatch");
    static_assert(sizeof(exclusive_push_constants)      == 16, "PushConstants size mismatch");

    inclusive_scan_push_constants inclusive_pc = {
        .input = input_span,
        .output = output_span,
        .groups = group_span
    };

    exclusive_push_constants exclusive_pc = {
        .groups = group_span
    };

    // Create command buffer
    auto commandPool = context.device.createCommandPoolUnique({
        {}, context.queueFamilyIndex});
    auto commandBuffers = context.device.allocateCommandBuffers({
        commandPool.get(), vk::CommandBufferLevel::ePrimary, 1});
    auto& cmd_buffer = commandBuffers.front();

    cmd_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    launch_computer_shader<inclusive_scan_push_constants>(cmd_buffer, inclusive_scan_shader, {NUM_GROUPS, 1, 1}, &inclusive_pc);

    auto barrier = vk::BufferMemoryBarrier2()
        .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
        .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
        .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
        .setBuffer(groupSumsBuffer.get())
        .setSize(VK_WHOLE_SIZE);

    cmd_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(barrier));

    launch_computer_shader<exclusive_push_constants>(cmd_buffer, exclusive_scan_object, {1, 1, 1}, &exclusive_pc);

    {
        std::array barriers = {
            vk::BufferMemoryBarrier2()
                .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderRead)
                .setBuffer(groupSumsBuffer.get())
                .setSize(VK_WHOLE_SIZE),
    
            vk::BufferMemoryBarrier2()
                .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite)
                .setBuffer(outputBuffer.get())
                .setSize(VK_WHOLE_SIZE)
        };
    
        cmd_buffer.pipelineBarrier2(vk::DependencyInfo().setBufferMemoryBarriers(barriers));
    }

    launch_computer_shader<inclusive_scan_push_constants>(cmd_buffer, propogate_scan_shader, {NUM_GROUPS, 1, 1}, &inclusive_pc);

    cmd_buffer.end();

    // Submit and wait
    auto fence = context.device.createFenceUnique({});
    context.queue.submit({vk::SubmitInfo({}, {}, cmd_buffer)}, fence.get());
    context.device.waitForFences(fence.get(), VK_TRUE, UINT64_MAX);

    auto* group_sums_data = static_cast<uint32_t*>(context.device.mapMemory(groupSumsMemory.get(), 0, BUFFER_SIZE));
    std::cout << "Group sums:" << std::endl;
    for (uint32_t i = 0; i < NUM_GROUPS; ++i)
        std::cout << group_sums_data[i] << " ";
    std::cout << std::endl;
    context.device.unmapMemory(groupSumsMemory.get());

    // Read results
    auto* outputData = static_cast<uint32_t*>(context.device.mapMemory(outputMemory.get(), 0, BUFFER_SIZE));
    std::cout << "Output:\n";
    for (uint32_t i = 0; i < BUFFER_COUNT; ++i)
        std::cout << outputData[i] << " ";
    std::cout << "\n";
    context.device.unmapMemory(outputMemory.get());
}

template<class... Ts> struct typelist{};

int main() {
    try {
        auto context = initializeVulkan();

        shader_manager manager(context.device, context.physical_device);

        shader_manager::device_info dev_info = manager.get_device_info();

        auto inclusive_scan_shader = manager.load_shader(
            "C:\\dev\\repos\\imgui_test\\shaders\\inclusive_scan.slang",
            "workgroup_inclusive_scan",
            {128, 1, 1}
        );
        
        auto exclusive_scan_shader = manager.load_shader(
            "C:\\dev\\repos\\imgui_test\\shaders\\inclusive_scan.slang",
            "subgroup_exclusive_scan",
            {dev_info.subgroup_size, 1, 1}
        );

        auto propogate_shader = manager.load_shader(
            "C:\\dev\\repos\\imgui_test\\shaders\\inclusive_scan.slang",
            "propogate_group_sums",
            {128, 1, 1}
        );

        launch_inclusive_scan(context, inclusive_scan_shader, exclusive_scan_shader, propogate_shader);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
} 