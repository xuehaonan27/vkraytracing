#ifndef RAYTRACER_HPP
#define RAYTRACER_HPP

#include "RayTrace/Application.hpp"
#include "Utils/ModelViewController.hpp"
#include "Utils/SceneList.hpp"
#include "Utils/UserSettings.hpp"

class RayTracer final: public RayTrace::Application {
  public:
    VULKAN_NON_COPIABLE(RayTracer)

    RayTracer(
        const UserSettings& userSettings,
        const Base::WindowConfig& windowConfig,
        VkPresentModeKHR presentMode
    );
    ~RayTracer();

  protected:
    const Base::Scene& GetScene() const override {
        return *scene_;
    }

    Base::UniformBufferObject GetUniformBufferObject(VkExtent2D extent) const override;

    void SetPhysicalDevice(
        VkPhysicalDevice physicalDevice,
        std::vector<const char*>& requiredExtensions,
        VkPhysicalDeviceFeatures& deviceFeatures,
        void* nextDeviceFeatures
    ) override;

    void OnDeviceSet() override;
    void CreateSwapChain() override;
    void DeleteSwapChain() override;
    void DrawFrame() override;
    void Render(VkCommandBuffer commandBuffer, size_t currentFrame, uint32_t imageIndex) override;

    void OnKey(int key, int scancode, int action, int mods) override;
    void OnCursorPosition(double xpos, double ypos) override;
    void OnMouseButton(int button, int action, int mods) override;
    void OnScroll(double xoffset, double yoffset) override;

  private:
    void LoadScene(uint32_t sceneIndex);
    void CheckAndUpdateBenchmarkState(double prevTime);
    void CheckFramebufferSize() const;

    uint32_t sceneIndex_ {};
    UserSettings userSettings_ {};
    UserSettings previousSettings_ {};
    SceneList::CameraInitialSate cameraInitialSate_ {};
    ModelViewController modelViewController_ {};

    std::unique_ptr<const Base::Scene> scene_;
    std::unique_ptr<class UserInterface> userInterface_;

    double time_ {};

    uint32_t totalNumberOfSamples_ {};
    uint32_t numberOfSamples_ {};
    bool resetAccumulation_ {};

    // Benchmark stats
    double sceneInitialTime_ {};
    double periodInitialTime_ {};
    uint32_t periodTotalFrames_ {};
};

#endif // RAY_TRACER
