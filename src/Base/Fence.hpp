#ifndef BASE_FENCE_HPP
#define BASE_FENCE_HPP

#include "Utils/Macros.hpp"

namespace Base {
class Device;

class Fence final {
  public:
    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence& operator=(Fence&&) = delete;

    explicit Fence(const Device& device, bool signaled);
    Fence(Fence&& other) noexcept;
    ~Fence();

    const class Device& Device() const {
        return device_;
    }

    const VkFence& Handle() const {
        return fence_;
    }

    void Reset();
    void Wait(uint64_t timeout) const;

  private:
    const class Device& device_;

    VkFence fence_ {};
};

} // namespace Base

#endif // BASE_FENCE_HPP
