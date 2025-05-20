#pragma once

struct queue {
  vk::Queue handle;

  uint32_t queue_family_index;
  uint32_t queue_index; // Index within family
};