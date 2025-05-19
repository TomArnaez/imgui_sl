#pragma once

#include <optional>

struct queue_family_properties {
  // Guaranteed to be one
  uint32_t queue_count;

  // If timestamps are supported, the number of bits supported by timestamp
  // operations. The returned value will be in the range 36..64.
  std::optional<uint32_t> timestamp_bits;
};

struct queue {
  vk::Queue handle;

  uint32_t queue_family_index;
  uint32_t queue_index; // Index within family
};