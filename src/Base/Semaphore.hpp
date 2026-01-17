#ifndef BASE_SEMAPHORE_HPP
#define BASE_SEMAPHORE_HPP

#include "Utils/Macros.hpp"

namespace Base {
class Device;

class Semaphore final {
  public:
    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;
    Semaphore& operator=(Semaphore&&) = delete;

    explicit Semaphore(const Device& device);
    Semaphore(Semaphore&& other) noexcept;
    ~Semaphore();

    const class Device& Device() const {
        return device_;
    }

  private:
    const class Device& device_;

    VULKAN_HANDLE(VkSemaphore, semaphore_)
};

} // namespace Base

#endif // BASE_SEMAPHORE_HPP
