#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include <slang-com-ptr.h>
#include <slang.h>
#include <cassert>

inline vk::DescriptorType map_slang_binding_type_to_vulkan_descriptor_type(slang::BindingType binding_type) {
    switch (binding_type) {
        case slang::BindingType::Sampler:           return vk::DescriptorType::eSampler;
        case slang::BindingType::Texture:           return vk::DescriptorType::eSampledImage;
        case slang::BindingType::ConstantBuffer:    return vk::DescriptorType::eUniformBuffer;
        case slang::BindingType::RawBuffer:         return vk::DescriptorType::eStorageBuffer;
        default:
            assert(false && "Unknown SlangStage");
            return static_cast<vk::DescriptorType>(-1);
    }
}

struct pipeline_layout_builder {
    std::vector<vk::DescriptorSetLayout> descriptor_set_layouts;
    std::vector<vk::PushConstantRange> push_constant_ranges;
    vk::ShaderStageFlagBits current_stage_flags;

    void add_descriptor_set_parameter_block(slang::TypeLayoutReflection* parameter_block_type_layout);
    void add_sub_object_ranges(slang::TypeLayoutReflection* type_layout);
    void add_sub_object_range(slang::TypeLayoutReflection* type_layout, uint32_t suboject_range_index);
    void add_push_constant_range_for_constant_buffer(slang::TypeLayoutReflection* constant_buffer_type_layout);
    vk::PipelineLayout finish(vk::Device device);
};

struct descriptor_set_layout_builder {
    std::vector<vk::DescriptorSetLayoutBinding> descriptor_ranges;

    void start(pipeline_layout_builder* pipeline_layout_builder);
    void add_global_scope_parameters(slang::ProgramLayout* program_layout);
    void add_entry_point_parameters(slang::ProgramLayout* program_layout);
    void add_entry_point_parameter(slang::EntryPointLayout* entry_point_layout);

    // If the the element type of the parameter block has any amount of ordinary data, then Slang compiler
    // will automatically introduce a uniform buffer to pass that data
    void add_ranges_for_parameter_block_element(slang::TypeLayoutReflection* element_type_layout);
    void add_ranges(slang::TypeLayoutReflection* type_layout);
    void add_descriptor_ranges(slang::TypeLayoutReflection* type_layout);
    void add_descriptor_range(
        slang::TypeLayoutReflection* type_layout,
        uint32_t relative_set_index,
        uint32_t range_index
    );
    void finish(vk::Device device);
private:
    void add_automatically_introduced_uniform_buffer();
    vk::ShaderStageFlagBits get_shader_stage_flags(SlangStage stage);

    vk::ShaderStageFlagBits current_stage_flags;
    pipeline_layout_builder* pipeline_layout_builder_;
    uint32_t set_index;
};

inline vk::PipelineLayout create_pipeline_layout(vk::Device device, slang::ProgramLayout* program_layout) {
    pipeline_layout_builder pipeline_layout_builder;
    descriptor_set_layout_builder descriptor_layout_builder;
    descriptor_layout_builder.start(&pipeline_layout_builder);

    descriptor_layout_builder.add_global_scope_parameters(program_layout);
    descriptor_layout_builder.add_entry_point_parameters(program_layout);

    descriptor_layout_builder.finish(device);
    return pipeline_layout_builder.finish(device);
}