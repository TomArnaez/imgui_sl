#pragma once

#include <bit>
#include <expected>
#include <format>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

/*========================================================================================
 *  slot_map<T, IndexBits, GenerationBits>
 *  -----------------------------------------------------------------------
 *  •  Dense, cache-friendly storage for trivially move-constructible types.
 *  •  32-bit handle packs <index, generation> for stale-handle detection.
 *=======================================================================================*/
template<
	typename T,
	size_t  IndexBits = 24,		//  2^24  ≅ 16.7 M live objects
	size_t  GenerationBits = 8>		//  2^8   = 256 generations / slot
class slot_map {
	static_assert(IndexBits + GenerationBits <= 32,
		"slot_map: IndexBits + GenerationBits must be ≤ 32");
	static_assert(std::is_move_constructible_v<T>,
		"slot_map: T must be move-constructible");
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

	private:
		friend slot_map;
		static constexpr id make(uint32_t idx, uint32_t gen) {
			return id{ static_cast<uint32_t>((gen << IndexBits) | idx) };
		}
	};

	slot_map() = default;
	slot_map(const slot_map&) = delete;
	slot_map& operator=(const slot_map&) = delete;

	template<typename... Args>
	[[nodiscard]]
	std::expected<id, std::string> emplace(Args&&... args) {
		uint32_t idx;
		uint32_t gen;

		if (free_.empty()) {
			idx = static_cast<uint32_t>(slots_.size());
			slots_.push_back(slot{});
			gen = 0;
		}
		else {
			idx = free_.back();
			free_.pop_back();
			gen = ++slots_[idx].generation;		// invalidate previous handles
		}

		try {
			new (&slots_[idx].storage)
				T(std::forward<Args>(args)...);
			slots_[idx].occupied = true;
			++live_;
		}
		catch (...) {
			free_.push_back(idx);					// roll-back
			return std::unexpected(
				"slot_map::emplace(): payload constructor threw");
		}
		return id::make(idx, gen);
	}

	[[nodiscard]]
	std::expected<const T*, std::string> get(id handle) const {
		return get_impl<const T>(handle);
	}

	[[nodiscard]]
	std::expected<T*, std::string> get(id handle) {
		return get_impl<T>(handle);
	}

	std::expected<std::monostate, std::string> remove(id handle) {
		auto slotp = get_impl<T>(handle);
		if (!slotp)
			return std::unexpected(slotp.error());

		std::destroy_at(slotp.value());

		slot& s = slots_[handle.index()];
		s.occupied = false;
		++s.generation;							// immediately stale-out old id
		free_.push_back(handle.index());
		--live_;
		return {};
	}

	[[nodiscard]] std::size_t size()  const noexcept { return live_; }
	[[nodiscard]] bool        empty() const noexcept { return live_ == 0; }

	template<typename Fn>
	void for_each(Fn&& fn) {
		for (auto& s : slots_)
			if (s.occupied)
				fn(*std::launder(reinterpret_cast<T*>(s.storage)));
	}

private:
	struct slot {
		alignas(T) std::byte storage[sizeof(T)];
		uint32_t			 generation = 0;
		bool				 occupied = false;

		T& payload() {
			return *std::launder(reinterpret_cast<T*>(storage));
		}
		const T& payload() const {
			return *std::launder(reinterpret_cast<const T*>(storage));
		}
	};

	/*  Shared implementation for const / non-const lookup  */
	template<typename U>
	std::expected<U*, std::string> get_impl(id handle) const {
		const uint32_t idx = handle.index();
		if (idx >= slots_.size())
			return std::unexpected("slot_map::get(): index out of range");

		const slot& s = slots_[idx];
		if (!s.occupied)
			return std::unexpected("slot_map::get(): slot empty");
		if (s.generation != handle.generation())
			return std::unexpected("slot_map::get(): stale handle");

		if constexpr (std::is_const_v<std::remove_pointer_t<U>>)
			return &s.payload();
		else
			return &const_cast<slot&>(s).payload();
	}

	std::vector<slot>      slots_;		// dense payload storage
	std::vector<uint32_t>  free_;		// free-list of vacant indices
	std::size_t            live_ = 0;	// occupied slot count
};
