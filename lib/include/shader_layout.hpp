#pragma once

#include <ranges>

namespace vkengine {

vk::DescriptorType map_slang_binding_type_to_vk(slang::BindingType binding_type) {
	switch (binding_type) {
	case slang::BindingType::Sampler: return vk::DescriptorType::eSampler;
	case slang::BindingType::Texture: return vk::DescriptorType::eSampledImage;
	default:
		assert(!"Unexpected binding type");
	}
}

vk::ShaderStageFlagBits map_slang_stage_to_vk(SlangStage stage) {
	switch (stage) {
	case SLANG_STAGE_COMPUTE: return vk::ShaderStageFlagBits::eCompute;
	default:
		assert(!"Unexepcted shader stage");
	}
}

class pipeline_layout_builder {
public:
	std::vector<vk::DescriptorSetLayout>	descriptor_set_layouts;
	std::vector<vk::PushConstantRange>		push_constant_ranges;
	std::reference_wrapper<vulkan_core>		vulkan_core_;

	pipeline_layout_builder(std::reference_wrapper<vulkan_core> vulkan_core) :
		vulkan_core_(vulkan_core) {
	}

	class descriptor_set_layout_builder {
	public:
		std::vector<vk::DescriptorSetLayoutBinding>		descriptor_binding_ranges;
		std::reference_wrapper<pipeline_layout_builder> pipeline_layout_builder_;
		std::reference_wrapper<vulkan_core>				vulkan_core_;
		uint32_t										set_index;

		descriptor_set_layout_builder(
			vulkan_core& vulkan_core,
			pipeline_layout_builder& pipeline_layout_builder
		)
			: vulkan_core_(vulkan_core), pipeline_layout_builder_(pipeline_layout_builder) {
			set_index = pipeline_layout_builder.descriptor_set_layouts.size();
			pipeline_layout_builder.descriptor_set_layouts.push_back(VK_NULL_HANDLE);
		}

		void add_automatically_introduced_uniforbuffer() {
			auto vulkan_binding_index = descriptor_binding_ranges.size();

			descriptor_binding_ranges.emplace_back(
				vk::DescriptorSetLayoutBinding()
				.setStageFlags(vk::ShaderStageFlagBits::eAll)
				.setBinding(vulkan_binding_index)
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			);
		}

		// In the common case, each of the range that is reflected by Slang translate to one range in the Vulkan descriptor set
		void add_descriptor_range(
			slang::TypeLayoutReflection* type_layout,
			uint32_t relative_set_index,
			uint32_t range_index,
			vk::ShaderStageFlagBits shader_stage
		) {
			slang::BindingType	binding_type = type_layout->getDescriptorSetDescriptorRangeType(relative_set_index, range_index);
			uint32_t			descriptor_count = type_layout->getDescriptorSetDescriptorRangeDescriptorCount(relative_set_index, range_index);
			uint32_t			binding_index = descriptor_binding_ranges.size();


			switch (binding_type) {
				// We account for push-constants elsewhere
			case slang::BindingType::PushConstant:
				return;
			default:
				break;
			}

			descriptor_binding_ranges.emplace_back(
				vk::DescriptorSetLayoutBinding()
				.setDescriptorCount(descriptor_count)
				.setDescriptorType(map_slang_binding_type_to_vk(binding_type))
				.setStageFlags(shader_stage)
			);
		}

		void add_descriptor_ranges(
			slang::TypeLayoutReflection* type_layout,
			vk::ShaderStageFlagBits shader_stage
		) {
			uint32_t relative_set_index = 0;
			uint32_t range_count = type_layout->getDescriptorSetDescriptorRangeCount(relative_set_index);

			for (uint32_t range_index = 0; range_index < range_count; ++range_index) {
				add_descriptor_range(type_layout, relative_set_index, range_index, shader_stage);
			}
		}

		void add_entry_point_parameters(slang::EntryPointLayout* entry_point_layout) {
			add_descriptor_ranges_for_parameter_block_element(
				entry_point_layout->getTypeLayout(),
				map_slang_stage_to_vk(entry_point_layout->getStage())
			);
		}

		void add_entry_point_parameters(
			slang::ProgramLayout* program_layout
		) {
			uint32_t entry_point_count = program_layout->getEntryPointCount();
			for (uint32_t entry_point_idx = 0; entry_point_idx < entry_point_count; ++entry_point_idx)
				add_entry_point_parameters(program_layout->getEntryPointByIndex(entry_point_idx));
		}

		void finish_building(
			pipeline_layout_builder& pipeline_layout_builder
		) {
			if (descriptor_binding_ranges.empty()) return;

			auto descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo()
				.setBindings(descriptor_binding_ranges);

			pipeline_layout_builder.descriptor_set_layouts[set_index] =
				vulkan_core_
				.get()
				.device()
				.createDescriptorSetLayout(descriptor_set_layout_create_info);
		}

		void add_descriptor_ranges_for_parameter_block_element(
			slang::TypeLayoutReflection* element_type_layout,
			vk::ShaderStageFlagBits shader_stage
		) {
			// If the element type of the parameter block (the Thing in ParameterBlock<Thing>) has any amount of ordinary data in it,
			// the Slang compiler automatically introduces a uniform buffer to pass that data. 
			// The automatically-introduced uniform buffer will only be present if it was needed (that is, when the element type has a non-zero size in bytes),
			// and it will always precede any other bindings for the parameter block.
			// https://shader-slang.org/docs/parameter-blocks/
			if (element_type_layout->getSize() > 0) add_automatically_introduced_uniforbuffer();

			add_descriptor_ranges(element_type_layout, shader_stage);
			pipeline_layout_builder_.get().add_sub_object_ranges(element_type_layout, shader_stage);
		}
	};

	void add_push_constant_range_for_constant_buffer(slang::TypeLayoutReflection* constant_buffer_type_layout, vk::ShaderStageFlagBits shader_stage) {
		slang::TypeLayoutReflection* element_type_layout = constant_buffer_type_layout->getElementTypeLayout();
		uint32_t element_size = element_type_layout->getSize();

		if (element_size == 0) return;

		push_constant_ranges.emplace_back(
			vk::PushConstantRange()
			.setStageFlags(shader_stage)
			.setOffset(0)
			.setSize(element_size)
		);
	}

	void add_sub_object_range(slang::TypeLayoutReflection* type_layout, uint32_t sub_object_range_index, vk::ShaderStageFlagBits shader_stage) {
		auto binding_range_idx = type_layout->getSubObjectRangeBindingRangeIndex(sub_object_range_index);
		slang::BindingType binding_type = type_layout->getBindingRangeType(binding_range_idx);

		switch (binding_type) {
		case slang::BindingType::ParameterBlock: {
			add_descriptor_set_parameter_block(type_layout->getBindingRangeLeafTypeLayout(binding_range_idx), shader_stage);
			break;
		}
		case slang::BindingType::PushConstant: {
			add_push_constant_range_for_constant_buffer(type_layout->getBindingRangeLeafTypeLayout(binding_range_idx), shader_stage);
			break;
		}
		default:
			return;
		}
	}

	void add_descriptor_set_parameter_block(
		slang::TypeLayoutReflection* parameter_block_type_layout,
		vk::ShaderStageFlagBits shader_stage
	) {
		descriptor_set_layout_builder descriptor_set_builder(vulkan_core_, *this);
		descriptor_set_builder.add_descriptor_ranges_for_parameter_block_element(parameter_block_type_layout->getElementTypeLayout(), shader_stage);
	}

	void add_sub_object_ranges(slang::TypeLayoutReflection* type_layout, vk::ShaderStageFlagBits shader_stage) {
		uint32_t sub_object_range_count = type_layout->getSubObjectRangeCount();

		for (uint32_t sub_object_range_idx = 0; sub_object_range_idx < sub_object_range_count; ++sub_object_range_idx)
			add_sub_object_range(type_layout, sub_object_range_idx, shader_stage);
	}

	void filter_out_empty_descriptor_set_layouts() {
		std::vector<vk::DescriptorSetLayout> filtered_descriptor_sets;
		for (auto& descriptor_set_layout : descriptor_set_layouts) {
			if (!descriptor_set_layout)
				continue;
			filtered_descriptor_sets.push_back(descriptor_set_layout);
		}
		std::swap(descriptor_set_layouts, filtered_descriptor_sets);
	}

	vk::PipelineLayout finish_building() {
		filter_out_empty_descriptor_set_layouts();

		return vulkan_core_.get().device().createPipelineLayout(
			vk::PipelineLayoutCreateInfo()
			.setSetLayouts(descriptor_set_layouts)
			.setPushConstantRanges(push_constant_ranges)
		);
	}
};

struct entry_point_layout {
	std::string name;
	vk::PipelineLayout pipeline_layout;
	std::vector<vk::PushConstantRange> push_constant_ranges;
};

struct shader_layout {
	std::vector<entry_point_layout> entry_point_layouts;
};

shader_layout create_pipeline_layout(slang::ProgramLayout* program_layout, std::reference_wrapper<vulkan_core> vulkan_core) {
	pipeline_layout_builder pipeline_layout_builder_(vulkan_core);
	pipeline_layout_builder::descriptor_set_layout_builder descriptor_set_layout_builder(vulkan_core, pipeline_layout_builder_);

	auto entry_point_layouts =
		std::views::iota(0u, program_layout->getEntryPointCount())
		| std::views::transform([&](auto idx) {
			auto* entry_point = program_layout->getEntryPointByIndex(idx);
			pipeline_layout_builder pipeline_builder(vulkan_core);
			pipeline_layout_builder::descriptor_set_layout_builder descriptor_set_layout_builder(vulkan_core, pipeline_builder);

			descriptor_set_layout_builder.add_entry_point_parameters(entry_point);
			descriptor_set_layout_builder.finish_building(pipeline_builder);
			vk::PipelineLayout pipeline_layout = pipeline_builder.finish_building();
			return entry_point_layout{ 
				.name = entry_point->getName(),
				.pipeline_layout = pipeline_layout,
				.push_constant_ranges = pipeline_builder.push_constant_ranges
			};
		});

	descriptor_set_layout_builder.add_entry_point_parameters(program_layout);
	descriptor_set_layout_builder.finish_building(pipeline_layout_builder_);
	vk::PipelineLayout pipeline_layout = pipeline_layout_builder_.finish_building();

	return shader_layout {
		.entry_point_layouts = entry_point_layouts | std::ranges::to<std::vector>()
	};
}
}