#pragma once

#include <vulkan_core.hpp>
#include <allocator.hpp>

namespace vkengine {

template<typename T>
struct device_span {
	vk::DeviceAddress span;
	uint32_t size;
};

template<typename T>
class typed_buffer {
public:
	typed_buffer(
		std::reference_wrapper<allocator> allocator, 
		const vulkan_core& core, 
		uint32_t element_count,
		VmaAllocationCreateInfo allocation_create_info = {}
	)
		: allocator(allocator), element_count(element_count) {
		buffer = allocator.get().create_buffer(
			vk::BufferCreateInfo()
				.setSize(sizeof(T) * element_count)
				.setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress),
			allocation_create_info
		);

		buffer_address = core.device().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer.handle));
	}

	void destroy() {
		allocator.get().destroy_buffer(buffer);
	}

	template<std::ranges::input_range R>
		requires std::same_as<std::ranges::range_value_t<R>, T>
	void copy_from_host(R&& src) {
		const auto source_size = std::ranges::size(src);

		if (source_size > element_count)
			throw detailed_exception("Source range size exceeds buffer capacity");

		T* mapped_data = mapping();
		std::ranges::copy(src, mapped_data);
	}

	device_span<T> device_span() const {
		return {
			.span = buffer_address,
			.size = element_count
		};
	}

	vk::DeviceAddress device_address() const {
		return buffer_address;
	}

	T* mapping() const {
		T* mapped_data = static_cast<T*>(buffer.allocation_info.pMappedData);
		if (!mapped_data)
			throw detailed_exception("Buffer not mapped to host memory");
		return mapped_data;
	}

	uint32_t size() const {
		return element_count;
	}

	uint32_t size_bytes() const {
		return buffer.size;
	}
private:
	std::reference_wrapper<allocator> allocator;
	vk::DeviceAddress buffer_address;
	buffer buffer;
	uint32_t element_count;
};

}