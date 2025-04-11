#pragma once

#include <slang.h>
#include <vulkan/vulkan_structs.hpp>

namespace vkengine {

struct binding_offset {
	// An offset in GLSL/SPIR-V "bindings"
	uint32_t binding = 0;
	
	// The descriptor set that the binding field indexes into
	uint32_t binding_set = 0;

	uint32_t push_constant_range_offset = 0;

	binding_offset() {}

	binding_offset(slang::VariableLayoutReflection* variable_layout) {
		binding_set = static_cast<uint32_t>(variable_layout->getBindingSpace(
			SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT);)
		binding =
			static_cast<uint32_t>(variable_layout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
		pushConstantRange =
			static_cast<uint32_t>(varLayout->getOffset(SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER));
	}
};

struct binding_range_info {
	slang::BindingType binding_type;
	uint32_t count;
	uint32_t base_index;

	// An index into the sub-object array if this binding range is treated as a sub-object
	uint32_t sub_object_index;

	// The "binding" offset to apply for this range
	uint32_t binding_offset;

	bool is_specialisable;
};

struct descriptor_set_info {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	vk::DescriptorSetLayout descriptor_set_layout;
};

struct entry_point_layout {
	slang::EntryPointLayout* layout_;
	vk::ShaderStageFlags shader_stage_flags_;
};

struct root_shader_object_layout {
	struct builder {
		void add_global_params(slang::VariableLayoutReflection* globals_layout) {
		}
	};

	slang::ProgramLayout*					program_layout_ = nullptr;
	std::vector<entry_point_layout>			entry_points_;
	vk::PipelineLayout						pipeline_layout_ = {};
	std::vector<vk::DescriptorSetLayout>	descriptor_set_layouts_;
	std::vector<vk::PushConstantRange>		push_constant_ranges_;
};

}