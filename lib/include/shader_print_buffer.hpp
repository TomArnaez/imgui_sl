#pragma once

#include <typed_buffer.hpp>

namespace vkengine {

enum print_op : uint16_t {
	new_line = 0,
	uint16,
	uint32
};

class shader_print_buffer {
	host_visible_buffer<uint32_t> print_buffer_;

	[[nodiscard]] uint32_t word_count() const {
		return print_buffer_.mapping()[0];
	}

	void clear_print_buffer() {
		print_buffer_.mapping()[0] = 0;
	}

	[[nodiscard]] std::string read_print_buffer() {
		uint32_t words = word_count();
		const uint32_t* buffer = print_buffer_.mapping();

		std::string out;

		uint32_t word_index = 1;
		while (word_index < words) {
			auto [lo, hi] = std::bit_cast<std::array<uint16_t, 2>>(buffer[word_index]);

			switch (auto op = static_cast<print_op>(hi); op) {
			default:
				assert(false && "bad print_op type");
			case print_op::new_line:
				out += "\n";
				break;
			case print_op::uint16:
				out += fmt::format("{}", static_cast<uint16_t>(buffer[word_index + 1]));
				break;
			case print_op::uint32:
				out += fmt::format("{}", static_cast<uint32_t>(buffer[word_index + 1]));
				break;
			}

			word_index += (1 + lo);
		}
	}
};

}