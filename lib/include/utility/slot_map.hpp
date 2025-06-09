#pragma once

#include <cassert>
#include <expected>
#include <format>
#include <limits>
#include <new>
#include <ranges>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

struct slot_id {
  // The number of low bits can that be used for user-tagged generations.
  constexpr static uint32_t TAG_BITS = 8;
  // The mask for user-tagged generations.
  constexpr static uint32_t TAG_MASK = (1 << TAG_BITS) - 1;

  // The number of bits following the user-tag bits that are used for the state
  // of a slot.
  constexpr static uint32_t STATE_BITS = 2;
  // The mask for state of a slot.
  constexpr static uint32_t STATE_MASK = 0b11 << TAG_BITS;

  // The state tag used to signify that the slot is vacant.
  constexpr static uint32_t VACANT_TAG = 0b00 << TAG_BITS;
  // The state tag used to signify that the slot is occupied.
  constexpr static uint32_t OCCUPIED_TAG = 0b01 << TAG_BITS;

  // The mask for the pure generation, excluding state and user tags.
  constexpr static uint32_t GENERATION_MASK =
      std::numeric_limits<uint32_t>::max() << (TAG_BITS + STATE_BITS);
  // The smallest value to increment the pure generation by.
  constexpr static uint32_t GENERATION_INC = 1u << (TAG_BITS + STATE_BITS);

  slot_id(uint32_t index, uint32_t generation)
    : index_(index), generation_(generation) {}

  [[nodiscard]]
  static bool is_occupied(uint32_t full_generation_field) {
    return (full_generation_field & STATE_MASK) == OCCUPIED_TAG;
  }

  [[nodiscard]]
  uint32_t tag() const {
    return generation_ & TAG_MASK;
  }

  [[nodiscard]]
  uint32_t index() const {
    return index_;
  }

  [[nodiscard]]
  uint32_t generation() const {
    return generation_;
  }

private:
  uint32_t index_ = 0;
  uint32_t generation_ = 0;
};

enum class slot_map_error : std::uint8_t {
    index_out_of_range,
    slot_empty,
    stale_handle,
    capacity_exhaused
  };

/*========================================================================================
 *  slot_map<T, IndexBits, GenerationBits>
 *  -----------------------------------------------------------------------
 *  •  Dense, cache-friendly storage for trivially copyable types.
 *  •  32-bit handle packs <index, generation> for stale-handle detection.
 *=======================================================================================*/
template <typename T, std::size_t MaxCapacity = (std::size_t{1} << 22)>
class slot_map {
  static_assert(std::is_nothrow_move_constructible_v<T>,
                "slot_map: T must be nothrow-move-constructible so that vector "
                "reallocation can move elements safely.");

public:
  using id = ::slot_id;

  slot_map() = default;
  slot_map(const slot_map &) = default;
  slot_map &operator=(const slot_map &) = default;

  ~slot_map() {
    for (auto &s : slots_)
      if (s.occupied)
        std::destroy_at(std::launder(reinterpret_cast<T *>(s.storage)));
  }

  // TODO: Return error if max capacity?
  template <typename... Args>
  [[nodiscard]]
  id emplace(Args &&...args) {
    uint32_t idx;

    if (free_.empty()) {
      assert(live_ >= MaxCapacity);

      idx = static_cast<uint32_t>(slots_.size());
      slots_.push_back(slot{});
    } else {
      idx = free_.back();
      free_.pop_back();
    }

    slot &s = slots_[idx];
    new (&s.storage) T(std::forward<Args>(args)...);

    s.generation = (s.generation & ~id::STATE_MASK) | id::OCCUPIED_TAG;
    ++live_;

    return id{idx, s.generation & ~id::STATE_MASK};
  }

  [[nodiscard]]
  std::optional<const T*> get(id handle) const {
    return get_impl<const T>(handle);
  }

  [[nodiscard]]
  std::optional<T*> get(id handle) {
    return get_impl<T>(handle);
  }

  [[nodiscard]]
  decltype(auto) get_unchecked(this auto &&self, std::uint32_t index) {
    return self.slots_[index].payload();
  }

  [[nodiscard]]
  std::optional<T> remove(id handle) {
    auto slotp = get_impl<T>(handle);
    if (slotp)
      return std::nullopt;

    // Safe: the slot is live and the pointer is valid.
    slot &s = slots_[handle.index()];
    T payload = s.payload();

    std::destroy_at(slotp.value());

    uint32_t pure_gen = s.generation & id::GENERATION_MASK;
    pure_gen += id::GENERATION_INC;
    if (pure_gen == 0)
      pure_gen = id::GENERATION_INC;

    s.generation = (s.generation & id::TAG_MASK) | pure_gen | id::VACANT_TAG;

    free_.push_back(handle.index());
    --live_;
    return payload;
  }

  [[nodiscard]] std::size_t size() const noexcept { return live_; }
  [[nodiscard]] bool        empty() const noexcept { return live_ == 0; }

  auto values() {
    namespace v = std::views;
    return v::iota(std::size_t{0}, slots_.size()) | v::filter([this](auto i) {
             return id::is_occupied(slots_[i].generation);
           }) |
           v::transform([this](auto i) -> T & { return slots_[i].payload(); });
  }

  auto values() const {
    namespace v = std::views;
    return v::iota(std::size_t{0}, slots_.size()) | v::filter([this](auto i) {
             return id::is_occupied(slots_[i].generation);
           }) |
           v::transform(
               [this](auto i) -> const T & { return slots_[i].payload(); });
  }

  auto entries() { return entries_impl(*this); }
  auto entries() const { return entries_impl(*this); }

  size_t constexpr capacity() const { return MaxCapacity; }

private:
  struct slot {
    alignas(T) std::byte storage[sizeof(T)];
    uint32_t generation = id::VACANT_TAG;

    T       &payload() { return *std::launder(reinterpret_cast<T *>(storage)); }
    const T &payload() const {
      return *std::launder(reinterpret_cast<const T *>(storage));
    }
  };

  template <typename Self>
  using payload_ref_t =
      std::conditional_t<std::is_const_v<Self>, const T &, T &>;

  template <typename Self> using entry_t = std::pair<id, payload_ref_t<Self>>;

  template <typename Self> static auto entries_impl(Self &self) {
    namespace v = std::views;

    return v::iota(std::size_t{0}, self.slots_.size()) |
           v::filter([&self](std::size_t i) {
             return id::is_occupied(self.slots_[i].generation);
           }) |
           v::transform([&self](std::size_t i) -> entry_t<Self> {
             auto &s = self.slots_[i];
             return {
                 id{static_cast<uint32_t>(i), s.generation & ~id::STATE_MASK},
                 s.payload()};
           });
  }

  /*  Shared implementation for const / non-const lookup  */
  template <typename U> std::optional<U*> get_impl(id handle) const {
    const uint32_t idx = handle.index();
    if (idx >= slots_.size())
      return std::nullopt;

    const slot &s = slots_[idx];
    if (!id::is_occupied(s.generation))
      return std::nullopt;

    if ((s.generation & ~id::STATE_MASK) != handle.generation())
      return std::nullopt;

    if constexpr (std::is_const_v<std::remove_pointer_t<U>>)
      return &s.payload();
    else
      return &const_cast<slot &>(s).payload();
  }

  std::vector<slot>     slots_;    // dense payload storage
  std::vector<uint32_t> free_;     // free-list of vacant indices
  std::size_t           live_ = 0; // occupied slot count
};
