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
	typed_buffer(std::reference_wrapper<allocator> allocator, const vulkan_core& core, uint32_t element_count)
		: allocator(allocator), element_count(element_count) {
		buffer = allocator.get().create_buffer(vk::BufferCreateInfo()
			.setSize(sizeof(T) * element_count)
			.setUsage(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress)
		);

		buffer_address = core.device().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(buffer.handle));
	}

	void destroy_buffer() {
		allocator.get().destroy_buffer(buffer);
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

	uint32_t size() const {
		return element_count;
	}

	uint32_t size_bytes() const {
		return element_count * sizeof(T);
	}

private:
	std::reference_wrapper<allocator> allocator;
	vk::DeviceAddress buffer_address;
	buffer buffer;
	uint32_t element_count;
};

}