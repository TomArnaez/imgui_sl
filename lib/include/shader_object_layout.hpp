#pragma once

#include <slang.h>
#include <unordered_map>
#include <vulkan/vulkan_structs.hpp>

namespace vkengine {

struct shader_layout_builder_base {
public:
	// Add any descriptor ranges implied by this object containing a leaf sub-object described by 'type_layout', at the given 'offset'
	template <typename Self>
	void add_descriptor_ranges_as_value(this Self&& self, slang::TypeLayoutReflection* type_layout, const binding_offset& offset) {
		// First we will scan through all the descriptor sets that the Slang reflection
		// information believes go into making up the given type.

		uint32_t descriptor_set_count = type_layout->getDescriptorSetCount();
		for (uint32_t i = 0; i < descriptor_set_count; ++i) {
			uint32_t descriptor_range_count = type_layout->getDescriptorSetDescriptorRangeCount(i);
			if (descriptor_range_count == 0)
				continue;

			uint32_t global_set_idx = offset.binding_set + type_layout->getDescriptorSetSpaceOffset(i);
			if (!descriptor_set_infos_.contains(global_set_idx))
				descriptor_set_infos_[global_set_idx] = descriptor_set_info{};
		}

		uint32_t binding_range_count = type_layout->getBindingRangeCount();
		for (uint32_t binding_range_idx = 0; binding_range_idx < binding_range_count; ++binding_range_idx) {
			auto binding_range_type = type_layout->getBindingRangeType(binding_range_idx);
			switch (binding_range_type) {
			default:
				break;

				// Skip over ranges that represent sub-objects, and handle them in a separate pass.
			case slang::BindingType::ParameterBlock:
				[[fallthrough]];
			case slang::BindingType::ConstantBuffer:
				[[fallthrough]];
			case slang::BindingType::ExistentialValue:
				[[fallthrough]];
			case slang::BindingType::PushConstant:
				continue;
			}

			// For a binding range we're interested in, enumerate over its contained descriptor ranges
			uint32_t descriptor_range_count = type_layout->getDescriptorSetDescriptorRangeCount(binding_range_idx);
			if (descriptor_range_count == 0)
				continue;

			uint32_t slang_descriptor_set_index = type_layout->getBindingRangeDescriptorSetIndex(binding_range_idx);
			descriptor_set_info& descriptor_set_info = descriptor_set_infos_[offset.binding_set + type_layout->getDescriptorSetSpaceOffset(slang_descriptor_set_index)];

			uint32_t first_descriptor_range_index = type_layout->getBindingRangeFirstDescriptorRangeIndex(binding_range_idx);
			for (uint32_t descriptor_range_index = first_descriptor_range_index; descriptor_range_index < first_descriptor_range_index + descriptor_range_count; ++descriptor_range_index) {
				auto slang_descriptor_type = type_layout->getDescriptorSetDescriptorRangeType(slang_descriptor_set_index, descriptor_range_index);

				// Certain kinds of descriptor ranges reflected by Slang do not
				// manifest as descriptors at the Vulkan level, so we will skip those
				switch (slang_descriptor_type) {
				case slang::BindingType::ParameterBlock:
					[[fallthrough]];
				case slang::BindingType::ConstantBuffer:
					[[fallthrough]];
				case slang::BindingType::ExistentialValue:
					continue;
				default:
					break;
				}

				auto vk_descriptor_type = map_descriptor_type(slang_descriptor_type);

				auto vk_binding_range_desc = vk::DescriptorSetLayoutBinding()
					.setBinding(type_layout->getDescriptorSetDescriptorRangeIndexOffset(slang_descriptor_set_index, descriptor_range_index))
					.setDescriptorCount(descriptor_range_count)
					.setDescriptorType(vk_descriptor_type)
					.setStageFlags(vk::ShaderStageFlagBits::eAll);

				descriptor_set_info.bindings.push_back(vk_binding_range_desc);
			}
		}

		// Now iterate over the sub-objects
		uint32_t sub_object_count = type_layout->getSubObjectRangeCount();
		for (uint32_t sub_object_range_index = 0; sub_object_range_index < sub_object_count; ++sub_object_range_index) {
			uint32_t binding_range_index = type_layout->getSubObjectRangeBindingRangeIndex(sub_object_range_index);

			auto binding_type = type_layout->getBindingRangeType(binding_range_index);
			auto sub_object_type_layout = type_layout->getBindingRangeLeafTypeLayout(binding_range_index);

			binding_offset sub_object_binding_offset = offset;
			sub_object_binding_offset += binding_offset(type_layout->getSubObjectRangeOffset(sub_object_range_index));

			auto handle_buffer = [this, sub_object_type_layout, sub_object_binding_offset](auto add_func) {
				auto container_var_layout = sub_object_type_layout->getContainerVarLayout();
				assert(container_var_layout != nullptr);

				auto element_var_layout = sub_object_type_layout->getElementVarLayout();
				assert(element_var_layout != nullptr);

				auto element_type_layout = element_var_layout->getTypeLayout();
				assert(element_type_layout != nullptr);

				binding_offset container_offset = sub_object_binding_offset;
				container_offset += binding_offset(container_var_layout);

				binding_offset element_offset = sub_object_binding_offset;
				element_offset += binding_offset(element_var_layout);

				std::invoke(add_func, this, element_type_layout, container_offset, element_offset);
				};

			switch (binding_type) {
			case slang::BindingType::ParameterBlock:
				// A ParameterBlock<X> never contributes descriptor ranges to the descriptor sets of the parent object
			default:
				break;
			case slang::BindingType::ExistentialValue:
				assert(false && "unsupported slang binding type");
				break;
			case slang::BindingType::ConstantBuffer:
				handle_buffer(&shader_object_layout::builder::add_descriptor_ranges_as_constant_buffer);
				break;
			case slang::BindingType::PushConstant:
				handle_buffer(&shader_object_layout::builder::add_descriptor_ranges_as_push_constant_buffer);
				break;
				break;
			}
		}
	}

	// Add the descriptor ranges defined by a 'ConstantBuffer<X>' where X is defined by 'element_type_layout'
	// The 'container_offset' and 'element_offset' are the binding offsets that should apply to the buffer itself and the contents of the buffer, respectively
	void add_descriptor_ranges_as_constant_buffer(
		slang::TypeLayoutReflection* element_type_layout,
		const binding_offset& container_offset,
		const binding_offset& element_offset
	) {
		// If the type has ordinary uniform data fields, we need to make sure to create
		// a descriptor set with a constant buffer binding in the case that the shader
		// object is bound as a stand alone parameter block
		if (element_type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM) != 0) {
			auto& descriptor_set = find_or_add_descriptor_set(container_offset.binding_set);
			descriptor_set.bindings.emplace_back(
				vk::DescriptorSetLayoutBinding()
				.setBinding(container_offset.binding)
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eUniformBuffer)
				.setStageFlags(vk::ShaderStageFlagBits::eAll)
			);
		}

		add_descriptor_ranges_as_value(element_type_layout, element_offset);
	}

	// Add the descriptor ranges implied by a 'PushConstantBuffer<X>' where 'X' is described by element_type_layout
	// The `containerOffset` and `elementOffset` are the binding offsets that should apply to the buffer itself and the contents of the buffer, respectively.
	void add_descriptor_ranges_as_push_constant_buffer(
		slang::TypeLayoutReflection* element_type_layout,
		const binding_offset& container_offset,
		const binding_offset& element_offset
	) {
		// If the type has ordinary uniform data fields, we need to make sure to create
		// a descriptor set with a constant buffer binding in the case that the shader
		// object is bound as a stand alone parameter block.
		auto ordinary_data_size = element_type_layout->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
		if (ordinary_data_size != 0) {
			uint32_t push_constant_range_index = container_offset.push_constant_range_offset;

			while (push_constant_ranges_.size() < push_constant_range_index)
				push_constant_ranges_.emplace_back(vk::PushConstantRange());

			push_constant_ranges_.emplace_back(vk::PushConstantRange()
				.setSize(ordinary_data_size)
				.setStageFlags(vk::ShaderStageFlagBits::eAll) // TODO: Be more clever
			);
		}

		add_descriptor_ranges_as_value(element_type_layout, element_offset);
	}

	descriptor_set_info& find_or_add_descriptor_set(uint32_t descriptor_set_index) {
		return descriptor_set_infos_.contains(descriptor_set_index) ?
			descriptor_set_infos_[descriptor_set_index] : descriptor_set_infos_.emplace(descriptor_set_index, descriptor_set_info{}).first->second;
	}

private:
	std::unordered_map<uint32_t, descriptor_set_info>	descriptor_set_infos_;
	std::vector<vk::PushConstantRange>					push_constant_ranges_;
};

vk::DescriptorType map_descriptor_type(slang::BindingType binding_type) {
	switch (binding_type) {
	case slang::BindingType::PushConstant:
	default:
		assert(false && "unsupported binding type");
		return static_cast<vk::DescriptorType>(-1);
	case slang::BindingType::Sampler:
		return vk::DescriptorType::eSampler;
	case slang::BindingType::CombinedTextureSampler:
		return vk::DescriptorType::eCombinedImageSampler;
	case slang::BindingType::Texture:
		return vk::DescriptorType::eSampledImage;
	case slang::BindingType::MutableTexture:
		return vk::DescriptorType::eStorageImage;
	case slang::BindingType::TypedBuffer:
		return vk::DescriptorType::eUniformTexelBuffer;
	case slang::BindingType::MutableTypedBuffer:
		return vk::DescriptorType::eStorageTexelBuffer;
	case slang::BindingType::RawBuffer:
	case slang::BindingType::MutableRawBuffer:
		return vk::DescriptorType::eStorageBuffer;
	case slang::BindingType::InputRenderTarget:
		return vk::DescriptorType::eInputAttachment;
	case slang::BindingType::InlineUniformData:
		return vk::DescriptorType::eInlineUniformBlockEXT;
	case slang::BindingType::RayTracingAccelerationStructure:
		return vk::DescriptorType::eAccelerationStructureKHR;
	case slang::BindingType::ConstantBuffer:
		return vk::DescriptorType::eUniformBuffer;
	}
}

struct binding_offset {
	// An offset in GLSL/SPIR-V "bindings"
	uint32_t binding = 0;
	
	// The descriptor set that the binding field indexes into
	uint32_t binding_set = 0;

	uint32_t push_constant_range_offset = 0;

	binding_offset() {}

	binding_offset(slang::VariableLayoutReflection* variable_layout) {
		binding_set					= static_cast<uint32_t>(variable_layout->getBindingSpace(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
		binding						= static_cast<uint32_t>(variable_layout->getOffset(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT));
		push_constant_range_offset	= static_cast<uint32_t>(variable_layout->getOffset(SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER));
	}

	void operator+=(binding_offset const& offset) {
		binding += offset.binding;
		binding_set += offset.binding_set;
		push_constant_range_offset += offset.push_constant_range_offset;
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
};

class shader_layout_builder_base {

private:
	std::vector<descriptor_set_info>	descriptor_set_infos_;
	std::vector<binding_range_info>		binding_range_infos_;
	std::vector<vk::PushConstantRange>	push_constant_ranges_;
	uint32_t							sub_object_count_;
};

class shader_object_layout {
public:

private:
	std::vector<descriptor_set_info>	descriptor_set_infos_;
	std::vector<binding_range_info>		binding_range_infos_;
	std::vector<vk::PushConstantRange>	push_constant_ranges_;
	uint32_t							sub_object_count_;
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