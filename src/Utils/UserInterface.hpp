#ifndef UTILS_USER_INTERFACE_HPP
#define UTILS_USER_INTERFACE_HPP

#include <memory>

#include "Macros.hpp"

namespace Base {
class CommandPool;
class DepthBuffer;
class DescriptorPool;
class FrameBuffer;
class RenderPass;
class SwapChain;
} // namespace

struct UserSettings;

struct Statistics final {
    VkExtent2D FramebufferSize;
    float FrameRate;
    float RayRate;
    uint32_t TotalSamples;
};

class UserInterface final {
  public:
    VULKAN_NON_COPIABLE(UserInterface)

    UserInterface(
        Base::CommandPool& commandPool,
        const Base::SwapChain& swapChain,
        const Base::DepthBuffer& depthBuffer,
        UserSettings& userSettings
    );
    ~UserInterface();

    void Render(
        VkCommandBuffer commandBuffer,
        const Base::FrameBuffer& frameBuffer,
        const Statistics& statistics
    );

    bool WantsToCaptureKeyboard() const;
    bool WantsToCaptureMouse() const;

    UserSettings& Settings() {
        return userSettings_;
    }

  private:
    void DrawSettings();
    void DrawOverlay(const Statistics& statistics);

    std::unique_ptr<Base::DescriptorPool> descriptorPool_;
    std::unique_ptr<Base::RenderPass> renderPass_;
    UserSettings& userSettings_;
};

#endif // UTILS_USER_INTERFACE_HPP
