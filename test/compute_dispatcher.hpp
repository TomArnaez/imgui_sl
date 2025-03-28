#pragma once

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

struct shader_object {
    vk::ShaderEXT shader_ext;
    vk::PipelineLayout pipeline_layout;
    vk::PushConstantRange push_constant_range;
    vk::ShaderStageFlagBits stage;
};

class shader_manager {
public:
    struct device_info {
        uint32_t subgroup_size;
        std::array<uint32_t, 3> max_compute_workgroup_sizes;
        uint32_t max_compute_workgroup_invocations;
        uint32_t max_compute_shared_memory_size;
    };

    shader_manager(vk::Device device, vk::PhysicalDevice physical_device)
        : device_(device) {

        device_info_ = {};

        auto chain = vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
        physical_device.getProperties2(&chain.get<vk::PhysicalDeviceProperties2>());

        const auto& properties2 = chain.get<vk::PhysicalDeviceProperties2>();
        const auto& limits = properties2.properties.limits;

        device_info_.subgroup_size = chain.get<vk::PhysicalDeviceSubgroupProperties>().subgroupSize;

        device_info_.max_compute_workgroup_sizes = {
            limits.maxComputeWorkGroupSize[0],
            limits.maxComputeWorkGroupSize[1],
            limits.maxComputeWorkGroupSize[1]
        };

        device_info_.max_compute_shared_memory_size = limits.maxComputeSharedMemorySize;

        slang::createGlobalSession(global_session_.writeRef());

        target_desc_.format = SLANG_SPIRV;
        target_desc_.profile = global_session_->findProfile("spirv_1_5");

        std::vector<slang::CompilerOptionEntry> compiler_option_entries = {
            {
                .name = slang::CompilerOptionName::VulkanUseEntryPointName,
                .value = { .kind = slang::CompilerOptionValueKind::Int, .intValue0 = 1}
            }
        };

        session_desc_.targets = &target_desc_;
        session_desc_.targetCount = 1;

        session_desc_.compilerOptionEntries = compiler_option_entries.data();
        session_desc_.compilerOptionEntryCount = compiler_option_entries.size();

        global_session_->createSession(session_desc_, session_.writeRef());

        std::string subgroup_module_src = std::format("export static const uint SUBGROUP_SIZE = {};",  device_info_.subgroup_size);
        subgroup_module_ = session_->loadModuleFromSourceString("subgroup_size", "subgroup_size.slang", subgroup_module_src.c_str());

        if (!subgroup_module_)
            throw std::runtime_error("Failed to create subgroup module");
    }

    device_info get_device_info() {
        return device_info_;
    }

    shader_object load_shader(
        const std::string& module_name,
        const std::string& entry_point_name,
        std::array<uint32_t, 3> workgroup_sizes) {
            std::string cache_key = std::format("{}:{}", module_name, entry_point_name);

            if (auto it = shader_cache_.find(cache_key); it != shader_cache_.end())
                return it->second;
            
            Slang::ComPtr<slang::IBlob> diagnostics = nullptr;

            Slang::ComPtr<slang::IModule> module(session_->loadModule(module_name.c_str(), diagnostics.writeRef()));

            if (!module)
                throw std::runtime_error("Failed to load module");

            Slang::ComPtr<slang::IEntryPoint> entry_point = nullptr;
            module->findEntryPointByName(entry_point_name.c_str(), entry_point.writeRef());

            if (!entry_point)
                std::runtime_error("Failed to load entry point");

            Slang::ComPtr<slang::IModule> workgroup_module = create_workgroup_module(workgroup_sizes);

            slang::IComponentType* program = nullptr;
            std::array<slang::IComponentType*, 4> components = { module, entry_point, subgroup_module_, workgroup_module };
            session_->createCompositeComponentType(components.data(), components.size(), &program, diagnostics.writeRef());

            if (!program)
                std::runtime_error("Failed to create slang program");
            
            slang::ProgramLayout* layout = program->getLayout();
            slang::EntryPointReflection* entry_point_reflection = layout->getEntryPointByIndex(0);

            vk::ShaderStageFlagBits stage;

            switch (entry_point_reflection->getStage()) {
            case SLANG_STAGE_COMPUTE:
                stage = vk::ShaderStageFlagBits::eCompute;
                break;
            default:
                throw std::runtime_error("Unsupported stage");
            }

            auto push_constants_range = vk::PushConstantRange().setStageFlags(stage);
            for (uint32_t param_idx = 0; param_idx < entry_point_reflection->getParameterCount(); ++param_idx) {
                auto param = entry_point_reflection->getParameterByIndex(param_idx);
                push_constants_range.size += param->getTypeLayout()->getSize();
            }

            Slang::ComPtr<slang::IBlob> spirv_code;
            program->getTargetCode(0, spirv_code.writeRef(), diagnostics.writeRef());

            if (!spirv_code)
                throw std::runtime_error("Failed to compile shader to SPIR-V");

            vk::ResultValue<vk::ShaderEXT> shader_ext = device_.createShaderEXT(
                vk::ShaderCreateInfoEXT()
                    .setStage(stage)
                    .setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
                    .setCodeSize(spirv_code->getBufferSize())
                    .setPName(entry_point_name.data())
                    .setPCode(static_cast<const uint32_t*>(spirv_code->getBufferPointer()))
                    .setPushConstantRanges(push_constants_range)
            );

            if (shader_ext.result != vk::Result::eSuccess)
                throw std::runtime_error("Failed to create shader object");

            shader_object shader_obj = {
                .shader_ext = shader_ext.value,
                .pipeline_layout = device_.createPipelineLayout(vk::PipelineLayoutCreateInfo().setPushConstantRanges(push_constants_range)),
                .push_constant_range = push_constants_range,
                .stage = stage
            };
            shader_cache_[cache_key] = shader_obj;

            return shader_obj;
        }
private:
    Slang::ComPtr<slang::IModule> create_workgroup_module(const std::array<uint32_t, 3>& workgroup_sizes) {
        std::string workgroup_module_src = std::format(
            "export static const uint WORKGROUP_SIZE_X = {};"
            "export static const uint WORKGROUP_SIZE_Y = {};"
            "export static const uint WORKGROUP_SIZE_Z = {};",
            workgroup_sizes[0],
            workgroup_sizes[1],
            workgroup_sizes[2]
        );

        Slang::ComPtr<slang::IModule> workgroup_module(session_->loadModuleFromSourceString("workgroup_sizes", "workgroup_sizes.slang", workgroup_module_src.c_str()));

        if (!workgroup_module)
            throw std::runtime_error("Failed to load workgroup module");

        return workgroup_module;
    }

    vk::Device device_;
    device_info device_info_;

    Slang::ComPtr<slang::IGlobalSession> global_session_;

    slang::SessionDesc session_desc_ = {};
    slang::TargetDesc target_desc_ = {};

    Slang::ComPtr<slang::ISession> session_;

    Slang::ComPtr<slang::IModule> subgroup_module_;

    std::unordered_map<std::string, shader_object> shader_cache_;
};