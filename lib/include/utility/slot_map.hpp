﻿#pragma once

#include <expected>
#include <format>
#include <new>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

/*========================================================================================
 *  slot_map<T, IndexBits, GenerationBits>
 *  -----------------------------------------------------------------------
 *  •  Dense, cache-friendly storage for trivially copyable types.
 *  •  32-bit handle packs <index, generation> for stale-handle detection.
 *=======================================================================================*/
template <typename T, std::size_t IndexBits = 24,
          std::size_t GenerationBits = 8,
          std::size_t MaxCapacity = (std::size_t{1} << IndexBits)>
class slot_map {
  static_assert(IndexBits + GenerationBits <= 32,
                "slot_map: IndexBits + GenerationBits must be ≤ 32");
  static_assert(MaxCapacity <= (1u << IndexBits),
                "MaxCapacity does not fit into IndexBits");
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "slot_map: T must be nothrow-move-constructible so that vector "
                "reallocation "
                "can move elements safely.");

public:
  struct id {
    uint32_t value = 0;

    [[nodiscard]]
    constexpr uint32_t index() const {
      return value & ((1u << IndexBits) - 1u);
    }

    [[nodiscard]]
    constexpr uint32_t generation() const {
      return value >> IndexBits;
    }

    friend auto operator<=>(id, id) = default;

    id(uint32_t v) : value(v) {}

  private:
    friend slot_map;
    static constexpr id make(uint32_t idx, uint32_t gen) {
      return id{static_cast<uint32_t>((gen << IndexBits) | idx)};
    }
  };

  enum class error : std::uint8_t {
    index_out_of_range,
    slot_empty,
    stale_handle,
    capacity_exhaused
  };

  slot_map() = default;
  slot_map(const slot_map &) = default;
  slot_map &operator=(const slot_map &) = default;

  ~slot_map() {
    for (auto &s : slots_)
      if (s.occupied)
        std::destroy_at(std::launder(reinterpret_cast<T *>(s.storage)));
  }

  template <typename... Args>
  [[nodiscard]]
  std::expected<id, error> emplace(Args &&...args) {
    uint32_t idx;

    if (free_.empty()) {
      if (live_ >= MaxCapacity)
        return std::unexpected(error::capacity_exhaused);

      idx = static_cast<uint32_t>(slots_.size());
      slots_.push_back(slot{});
    } else {
      idx = free_.back();
      free_.pop_back();
    }

    slot &s = slots_[idx];
    new (&s.storage) T(std::forward<Args>(args)...);

    s.occupied = true;
    ++live_;

    return id::make(idx, s.generation & generation_mask);
  }

  [[nodiscard]]
  std::expected<const T *, error> get(id handle) const {
    return get_impl<const T>(handle);
  }

  [[nodiscard]]
  std::expected<T *, error> get(id handle) {
    return get_impl<T>(handle);
  }

  [[nodiscard]]
  decltype(auto) get_unchecked(this auto &&self, std::uint32_t index) {
    return self.slots_[index].payload();
  }

  [[nodiscard]]
  std::expected<std::monostate, error> remove(id handle) {
    // First ask the common helper to validate the handle.
    auto slotp = get_impl<T>(handle);
    if (!slotp)
      return std::unexpected(slotp.error());

    // Safe: the slot is live and the pointer is valid.
    slot &s = slots_[handle.index()];

    std::destroy_at(slotp.value());
    s.occupied = false;
    s.generation = (s.generation + 1) & generation_mask;

    free_.push_back(handle.index());
    --live_;
    return {};
  }

  [[nodiscard]] std::size_t size() const noexcept { return live_; }
  [[nodiscard]] bool        empty() const noexcept { return live_ == 0; }

  auto values() {
    namespace v = std::views;
    return v::iota(std::size_t{0}, slots_.size()) |
           v::filter([this](auto i) { return slots_[i].occupied; }) |
           v::transform([this](auto i) -> T & { return slots_[i].payload(); });
  }

  auto values() const {
    namespace v = std::views;
    return v::iota(std::size_t{0}, slots_.size()) |
           v::filter([this](auto i) { return slots_[i].occupied; }) |
           v::transform(
               [this](auto i) -> const T & { return slots_[i].payload(); });
  }

  auto entries() { return entries_impl(*this); }
  auto entries() const { return entries_impl(*this); }

  size_t constexpr capacity() const { return MaxCapacity; }

private:
  struct slot {
    alignas(T) std::byte storage[sizeof(T)];
    uint32_t generation = 0;
    bool     occupied = false;

    T       &payload() { return *std::launder(reinterpret_cast<T *>(storage)); }
    const T &payload() const {
      return *std::launder(reinterpret_cast<const T *>(storage));
    }
  };

  template <typename Self>
  using payload_ref_t =
      std::conditional_t<std::is_const_v<Self>, const T &, T &>;

  template <typename Self> using entry_t = std::pair<id, payload_ref_t<Self>>;

  /*  Shared implementation for const / non-const entries range  */
  template <typename Self> static auto entries_impl(Self &self) {
    namespace v = std::views;

    return v::iota(std::size_t{0}, self.slots_.size()) |
           v::filter(
               [&self](std::size_t i) { return self.slots_[i].occupied; }) |
           v::transform([&self](std::size_t i) -> entry_t<Self> {
             auto &s = self.slots_[i]; // constness honoured
             return {id::make(static_cast<uint32_t>(i),
                              s.generation & generation_mask),
                     s.payload()}; // T&  or const T&
           });
  }

  /*  Shared implementation for const / non-const lookup  */
  template <typename U> std::expected<U *, error> get_impl(id handle) const {
    const uint32_t idx = handle.index();
    if (idx >= slots_.size())
      return std::unexpected(error::index_out_of_range);

    const slot &s = slots_[idx];
    if (!s.occupied)
      return std::unexpected(error::slot_empty);
    if (s.generation != handle.generation())
      return std::unexpected(error::stale_handle);

    if constexpr (std::is_const_v<std::remove_pointer_t<U>>)
      return &s.payload();
    else
      return &const_cast<slot &>(s).payload();
  }

  static constexpr uint32_t generation_mask = (1u << GenerationBits) - 1u;

  std::vector<slot>     slots_;    // dense payload storage
  std::vector<uint32_t> free_;     // free-list of vacant indices
  std::size_t           live_ = 0; // occupied slot count
};
