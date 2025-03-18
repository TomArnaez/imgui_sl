#include "pipeline_layout_builder.hpp"

void pipeline_layout_builder::add_descriptor_set_parameter_block(slang::TypeLayoutReflection* parameter_block_type_layout
) {
    descriptor_set_layout_builder desc_set_layout_builder;
}

void pipeline_layout_builder::add_sub_object_ranges(
    slang::TypeLayoutReflection* type_layout
) {
    uint32_t sub_object_range_count = type_layout->getSubObjectRangeCount();
    for (uint32_t sub_object_range_index = 0; sub_object_range_index < sub_object_range_count; ++sub_object_range_index)
        add_sub_object_range(type_layout, sub_object_range_index);
}

void pipeline_layout_builder::add_sub_object_range(
    slang::TypeLayoutReflection* type_layout,
    uint32_t suboject_range_index
) {
    auto binding_range_index = type_layout->getSubObjectRangeBindingRangeIndex(suboject_range_index);
    slang::BindingType binding_type = type_layout->getBindingRangeType(binding_range_index);
    switch (binding_type) {
    case slang::BindingType::ParameterBlock: {
        auto parameter_block_type_layout = type_layout->getBindingRangeLeafTypeLayout(binding_range_index);
        add_descriptor_set_parameter_block(parameter_block_type_layout);
    }
    case slang::BindingType::PushConstant: {
        auto constant_buffer_type_layout = type_layout->getBindingRangeLeafTypeLayout(binding_range_index);
        add_push_constant_range_for_constant_buffer(constant_buffer_type_layout);
    }
    default: 
        return;
    }
}

void pipeline_layout_builder::add_push_constant_range_for_constant_buffer(
    slang::TypeLayoutReflection* constant_buffer_type_layout
) {
    auto element_type_layout = constant_buffer_type_layout->getElementTypeLayout();
    auto element_size = element_type_layout->getSize();

    if (element_size == 0) return;

    push_constant_ranges.push_back(
        vk::PushConstantRange()
            .setStageFlags(current_stage_flags)
            .setSize(element_size));
}

vk::PipelineLayout pipeline_layout_builder::finish(
    vk::Device device
) {
    std::vector<vk::DescriptorSetLayout> filtered_descriptor_set_layouts;
    for (auto& layout : descriptor_set_layouts) {
        if (!layout)
            continue;
        filtered_descriptor_set_layouts.push_back(layout);
    }

    std::swap(descriptor_set_layouts, filtered_descriptor_set_layouts);

    return device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo()
            .setSetLayouts(descriptor_set_layouts)
            .setPushConstantRanges(push_constant_ranges));
}

void descriptor_set_layout_builder::start(pipeline_layout_builder* in_layout) {
    pipeline_layout_builder_ = in_layout;
    set_index = pipeline_layout_builder_->descriptor_set_layouts.size();
    pipeline_layout_builder_->descriptor_set_layouts.push_back({});
}

void descriptor_set_layout_builder::add_global_scope_parameters(slang::ProgramLayout* program_layout) {
    current_stage_flags = vk::ShaderStageFlagBits::eAll;
    add_ranges_for_parameter_block_element(program_layout->getGlobalParamsTypeLayout());
}

void descriptor_set_layout_builder::add_entry_point_parameters(slang::ProgramLayout* program_layout) {
    uint32_t entry_point_count = program_layout->getEntryPointCount();
    for (uint32_t entry_point_idx = 0; entry_point_idx < entry_point_count; ++entry_point_idx) {
        add_entry_point_parameter(program_layout->getEntryPointByIndex(entry_point_idx));
    }
}

void descriptor_set_layout_builder::add_entry_point_parameter(slang::EntryPointLayout* entry_point_layout) {
    current_stage_flags = get_shader_stage_flags(entry_point_layout->getStage());
    add_ranges_for_parameter_block_element(entry_point_layout->getTypeLayout());
}

// If the the element type of the parameter block has any amount of ordinary data, then Slang compiler
// will automatically introduce a uniform buffer to pass that data
void descriptor_set_layout_builder::add_ranges_for_parameter_block_element(slang::TypeLayoutReflection* element_type_layout) {
    if (element_type_layout->getSize() > 0)
        add_automatically_introduced_uniform_buffer();

    add_ranges(element_type_layout);
}

void descriptor_set_layout_builder::add_ranges(slang::TypeLayoutReflection* type_layout) {
    add_descriptor_ranges(type_layout);
    pipeline_layout_builder_->add_sub_object_ranges(type_layout);
}

void descriptor_set_layout_builder::add_descriptor_ranges(slang::TypeLayoutReflection* type_layout) {
    uint32_t relative_set_index = 0;
    uint32_t range_count = type_layout->getDescriptorSetDescriptorRangeCount(relative_set_index);

    for (uint32_t range_index = 0; range_index < range_count; ++range_index)
        add_descriptor_range(type_layout, relative_set_index, range_index);
}

void descriptor_set_layout_builder::add_descriptor_range(
    slang::TypeLayoutReflection* type_layout,
    uint32_t relative_set_index,
    uint32_t range_index
) {
    slang::BindingType binding_type = type_layout->getDescriptorSetDescriptorRangeType(relative_set_index, range_index);

    // Some ranges need to be skipped
    if (binding_type == slang::BindingType::PushConstant)
        return;

    auto descriptor_count = type_layout->getDescriptorSetDescriptorRangeDescriptorCount(relative_set_index, range_index);
    auto binding_index = descriptor_ranges.size();
    auto binding_range = vk::DescriptorSetLayoutBinding()
        .setBinding(binding_index)
        .setDescriptorCount(descriptor_count)
        .setDescriptorType(map_slang_binding_type_to_vulkan_descriptor_type(binding_type))
        .setStageFlags(current_stage_flags);

    descriptor_ranges.push_back(binding_range);
}

void descriptor_set_layout_builder::finish(vk::Device device) {
    // It's possible to have a ParameterBlock that contains nothing but other blocks, and there are no descriptor ranges in the outer block
    // Thus it is irrelevant
    if (descriptor_ranges.size() == 0)
        return;
    
    pipeline_layout_builder_->descriptor_set_layouts[set_index] = 
        device.createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo()
                .setBindings(descriptor_ranges));
}

void descriptor_set_layout_builder::add_automatically_introduced_uniform_buffer() {
    descriptor_ranges.push_back(
        vk::DescriptorSetLayoutBinding()
            .setStageFlags(vk::ShaderStageFlagBits::eAll)
            .setBinding(descriptor_ranges.size())
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eUniformBuffer));
}

vk::ShaderStageFlagBits descriptor_set_layout_builder::get_shader_stage_flags(SlangStage stage) {
    switch (stage) {
    case SLANG_STAGE_VERTEX:   return vk::ShaderStageFlagBits::eVertex;
    case SLANG_STAGE_FRAGMENT: return vk::ShaderStageFlagBits::eFragment;
    case SLANG_STAGE_COMPUTE:  return vk::ShaderStageFlagBits::eCompute;
    default:
        assert(false && "Unknown SlangStage");
        return static_cast<vk::ShaderStageFlagBits>(0);
    }
}