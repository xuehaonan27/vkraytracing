#ifndef BASE_RENDER_PASS_HPP
#define BASE_RENDER_PASS_HPP

#include "DepthBuffer.hpp"
#include "SwapChain.hpp"

namespace Base {
class RenderPass final {
  public:
    VULKAN_NON_COPIABLE(RenderPass)

    RenderPass(
        const SwapChain& swapChain,
        const DepthBuffer& depthBuffer,
        VkAttachmentLoadOp colorBufferLoadOp,
        VkAttachmentLoadOp depthBufferLoadOp
    );
    ~RenderPass();

    const class SwapChain& SwapChain() const {
        return swapChain_;
    }

    const class DepthBuffer& DepthBuffer() const {
        return depthBuffer_;
    }

  private:
    const class SwapChain& swapChain_;
    const class DepthBuffer& depthBuffer_;

    VULKAN_HANDLE(VkRenderPass, renderPass_)
};

} // namespace Base

#endif // BASE_RENDER_PASS_HPP
