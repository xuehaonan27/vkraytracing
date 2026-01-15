#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<char const*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif // NDEBUG

static std::vector<char> readFile(const std::string& filename) {
  // start read at the end of the file
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  std::vector<char> buffer(file.tellg());

  // Seek back to the beginning of the file
  file.seekg(0, std::ios::beg);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  file.close();

  return buffer;
}

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow* window;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  // queues are implicitly cleaned up when the device is destroyed, no need for cleanup
  uint32_t queueIndex = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> swapChainImages;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;
  vk::Format swapChainImageFormat = vk::Format::eUndefined;
  vk::Extent2D swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;

  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::CommandPool commandPool = nullptr;
  vk::raii::CommandBuffer commandBuffer = nullptr;

  // Semaphores to synchronize GPU operations
  vk::raii::Semaphore presentCompleteSemaphore = nullptr;
  vk::raii::Semaphore renderFinishedSemaphore = nullptr;
  // Fence to synchronize operations between GPU and CPU
  vk::raii::Fence drawFence = nullptr;

  const std::vector<const char*> requiredDeviceExtension = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRShaderDrawParametersExtensionName
  };

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "vkraytracing", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }

    device.waitIdle();
  }

  void cleanup() {
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  void createInstance() {
    constexpr vk::ApplicationInfo appInfo {
        .pApplicationName = "Hello Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion14
    };

    // Get the required layers
    std::vector<char const*> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation
    auto layerProperties = context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
          return std::ranges::none_of(layerProperties, [requiredLayer](auto const& layerProperty) {
            { return strcmp(layerProperty.layerName, requiredLayer) == 0; }
          });
        })) {
      throw std::runtime_error("One or more required layers are not supported!");
    }

    auto requiredExtensions = getRequiredExtensions();
    checkRequiredExtensions(requiredExtensions);

    vk::InstanceCreateInfo createInfo {
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data()
    };

    instance = vk::raii::Instance(context, createInfo);
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
    );
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
        | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
        | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    );
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT {
        .messageSeverity = severityFlags,
        .messageType = messageTypeFlags,
        .pfnUserCallback = &debugCallback
    };
    debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
      vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
      vk::DebugUtilsMessageTypeFlagsEXT type,
      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
      void*
  ) {
    std::cerr << "validation layer: type " << vk::to_string(type)
              << "msg: " << pCallbackData->pMessage << std::endl;
    return vk::False;
  }

  // Get the required instance extensions
  std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    return extensions;
  }

  // Check required extensions
  void checkRequiredExtensions(std::vector<const char*>& extensions) {
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    std::cout << "available extensions:\n";
    for (const auto& extension : extensionProperties) {
      std::cout << '\t' << extension.extensionName << '\n';
    }

    std::cout << "needed extensions:\n";
    for (const auto ext : extensions) {
      std::cout << '\t' << std::string(ext) << '\n';
      if (std::ranges::none_of(extensionProperties, [ext](auto const& extensionProperty) {
            return strcmp(extensionProperty.extensionName, ext) == 0;
          })) {
        throw std::runtime_error("Required GLFW extension not supported: " + std::string(ext));
      }
    }
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  void pickPhysicalDevice() {
    std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    const auto devIter = std::ranges::find_if(devices, [&](auto const& device) {
      // Check if the device supports the Vulkan 1.3 API version
      bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

      // Check if any of the queue families support graphics operations
      auto queueFamilies = device.getQueueFamilyProperties();
      bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
        return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
      });

      // Check if all required device extensions are available
      auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
      bool supportsAllRequiredExtensions = std::ranges::all_of(
          requiredDeviceExtension,
          [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
            return std::ranges::any_of(
                availableDeviceExtensions,
                [requiredDeviceExtension](auto const& availableDeviceExtension) {
                  return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension)
                      == 0;
                }
            );
          }
      );

      auto features = device.template getFeatures2<
          vk::PhysicalDeviceFeatures2,
          vk::PhysicalDeviceVulkan13Features,
          vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
      bool supportsRequiredFeatures =
          features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering
          && features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
                 .extendedDynamicState;

      return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions
          && supportsRequiredFeatures;
    });
    if (devIter != devices.end()) {
      physicalDevice = *devIter;
    } else {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  bool isDeviceSuitable(vk::raii::PhysicalDevice device) {
    auto deviceProperties = device.getProperties();
    auto deviceFeatures = device.getFeatures();

    // TODO: more details like device memory and queue families
    if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
        && deviceFeatures.geometryShader) {
      return true;
    }
    return false;
  }

  void createLogicalDevice() {
    uint32_t qindex = findQueueFamilies(physicalDevice);

    // // Specify the set of device features we will use
    // vk::PhysicalDeviceFeatures deviceFeatures;

    // Create a chain of feature structures
    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        featureChain = {
            // vk::PhysicalDeviceFeatures2 (empty for now)
            {},
            // Enable dynamic rendering from Vulkan 1.3
            {.synchronization2 = true, .dynamicRendering = true},
            // Enable extended dynamic state from the extension
            {.extendedDynamicState = true}
        };

    // This structure describes the number of queues we want for a single queue family
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo {
        .queueFamilyIndex = qindex, // graphics capabilities
        .queueCount = 1, // 1 queue per family
        .pQueuePriorities = &queuePriority // required even only 1 queue
    };

    vk::DeviceCreateInfo deviceCreateInfo {
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
        .ppEnabledExtensionNames = requiredDeviceExtension.data(),
    };

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    // Logical device, queue family index, queue index in the family
    queueIndex = qindex;
    queue = vk::raii::Queue(device, qindex, 0);
  }

  uint32_t findQueueFamilies(vk::raii::PhysicalDevice physicalDevice) {
    // find the index of the first queue family that supports graphics
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both graphics and present
    uint32_t queueIndex = ~0;
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics)
          && physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        // found a queue family that supports both graphics and present
        queueIndex = qfpIndex;
        break;
      }
    }
    if (queueIndex == ~0) {
      throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    return queueIndex;
  }

  void createSwapChain() {
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
    swapChainImageFormat = swapChainSurfaceFormat.format;

    vk::SwapchainCreateInfoKHR swapChainCreateInfo {
        .surface = *surface,
        .minImageCount = chooseSwapMinImageCount(surfaceCapabilities),
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        // eExclusive if graphics and present queue families are the same, which is usually the case
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)),
        .clipped = true
    };

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
  }

  static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount)
        && (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  static vk::SurfaceFormatKHR
  chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats) {
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(availableFormats, [](const auto& format) {
      return format.format == vk::Format::eB8G8R8A8Srgb
          && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  static vk::PresentModeKHR
  chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) {
      return presentMode == vk::PresentModeKHR::eFifo;
    }));
    return std::ranges::any_of(
               availablePresentModes,
               [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }
           )
        ? vk::PresentModeKHR::eMailbox
        : vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != 0xFFFFFFFF) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(
            width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width
        ),
        std::clamp<uint32_t>(
            height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height
        )
    };
  }

  void createImageViews() {
    swapChainImageViews.clear();

    vk::ImageViewCreateInfo imageViewCreateInfo {
        .viewType = vk::ImageViewType::e2D,
        .format = swapChainImageFormat,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0, // without any mipmapping levels
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    for (auto& image : swapChainImages) {
      imageViewCreateInfo.image = image;
      // Here the constructor of `vk::raii::ImageView` borrows `imageViewCreateInfo`, not moving
      swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
  }

  void createGraphicsPipeline() {
    auto shaderCode = readFile("shaders/shader.spv");
    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = shaderModule,
        .pName = "vertMain",
        .pSpecializationInfo = nullptr, // shader compile time constants assignment
    };

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = shaderModule,
        .pName = "fragMain",
        .pSpecializationInfo = nullptr, // shader compile time constants assignment
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Vertex input
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

    // Input assembly
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly {
        .topology = vk::PrimitiveTopology::eTriangleList
    };

    // Dynamic states
    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState {
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()
    };

    // Viewports and scissors
    vk::PipelineViewportStateCreateInfo viewportState {
        .viewportCount = 1,
        .pViewports = nullptr, // Set statically here if dynamic states disabled
        .scissorCount = 1,
        .pScissors = nullptr // Set statically here if dynamic states disabled
    };

    // Rasterizer
    vk::PipelineRasterizationStateCreateInfo rasterizer {
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasSlopeFactor = 1.0f,
        .lineWidth = 1.0f
    };

    // Multisamlping
    // One of the ways to perform antialiasing. GPU feature needed.
    vk::PipelineMultisampleStateCreateInfo multisampling {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False
    };

    // Depth and stencil testing
    // vk::PipelineDepthStencilStateCreateInfo;

    // Color blending
    vk::PipelineColorBlendAttachmentState colorBlendAttachment {
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
            | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    // Alpha blending: the new color is to be blended with the old color based on its opacity.
    /*
    colorBlendAttachment.blendEnable = vk::True;
    colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
    colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    */

    // references the array of structures for all the framebuffers and allows you to set blend
    // constants that you can use as blend factors in the aforementioned calculations
    vk::PipelineColorBlendStateCreateInfo colorBlending {
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    // Pipeline layout
    // Globals similar to dynamic state variables that can be changed at drawing time to alter
    // the behavior of shaders without having to recreate them.
    // Commonly used to pass the transformation matrix to the vertex shader, or to create texture
    // samplers in the fragment shader.
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0
    };
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Pipeline rendering color
    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo {
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapChainImageFormat
    };

    // Create the pipeline
    vk::GraphicsPipelineCreateInfo pipelineInfo {
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = nullptr, // because using dynamic rendering instead of traditional render pass
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1
    };

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
  }

  [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
    vk::ShaderModuleCreateInfo createInfo {
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };
    vk::raii::ShaderModule shaderModule {device, createInfo};
    return shaderModule;
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo {
        // will record a command buffer every frame, so must be able to rest and rerecord over it.
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex
    };

    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createCommandBuffer() {
    vk::CommandBufferAllocateInfo allocInfo {
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    vk::CommandBufferBeginInfo commandBufferBeginInfo {};
    commandBuffer.begin(commandBufferBeginInfo);

    // With dynamic rendering, no need for creating a render pass or framebuffers
    // Instead, specify the attachments directly when begin rendering
    // Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite, // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStage
    );

    // Set up the color attachment
    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::RenderingAttachmentInfo attachmentInfo = {
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, // the layout during rendering
        .loadOp = vk::AttachmentLoadOp::eClear, // what to do with the image before rendering
        .storeOp = vk::AttachmentStoreOp::eStore, // what to do with the image after rendering
        .clearValue = clearColor
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        .layerCount = 1, // the number of layers to reader to, which is 1 for non-layered image
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);

    // Bind the graphics pipeline
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    // Because using dynamic rendering, specify viewport and scissor state for this pipeline now
    // Viewports and scissors
    vk::Viewport viewport {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };
    vk::Rect2D scissor {vk::Offset2D {0, 0}, swapChainExtent};
    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0, scissor);

    // Issue the draw command for the triangle
    // (vertexCount, instanceCount, firstVertex, firstInstance)
    commandBuffer.draw(3, 1, 0, 0);

    // Finishing up
    commandBuffer.endRendering();

    // After rendering, transition the image layout back so it can be presented to the screen
    transition_image_layout(
        imageIndex,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe
    );

    commandBuffer.end();
  }

  // Transition the image layout to one that is suitable for rendering
  void transition_image_layout(
      uint32_t imageIndex,
      vk::ImageLayout oldLayout,
      vk::ImageLayout newLayout,
      vk::AccessFlags2 srcAccessMask,
      vk::AccessFlags2 dstAccessMask,
      vk::PipelineStageFlags2 srcStageMask,
      vk::PipelineStageFlags2 dstStageMask
  ) {
    vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapChainImages[imageIndex],
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };
    vk::DependencyInfo dependencyInfo =
        {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    commandBuffer.pipelineBarrier2(dependencyInfo);
  }

  // Steps:
  // 1. Wait for the previous frame to finish
  // 2. Acquire an image from the swap chain
  // 3. Record a command buffer which draws the scene onto that image
  // 4. Submit the recorded command buffer
  // 5. Present the swap chain image
  void drawFrame() {
    // 1. Wait until the previous frame has finished
    // Until then could CPU issue operations for the next frame
    // auto fenceResult = device.waitForFences(*drawFence, vk::True, UINT64_MAX);

    // 2. Acquire an image from the swap chain
    auto [result, imageIndex] =
        swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore, nullptr);

    // 3. Record a command buffer which draws the scene onto that image
    recordCommandBuffer(imageIndex);
    device.resetFences(*drawFence);

    // 4. Submit the recorded command buffer
    vk::PipelineStageFlags waitDestinationStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput
    );
    const vk::SubmitInfo submitInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphore,
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphore
    };
    queue.submit(submitInfo, *drawFence);

    // 1. Wait until the previous frame has finished
    // Until then could CPU issue operations for the next frame
    auto fenceResult = device.waitForFences(*drawFence, vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("failed to wait for fence!");
    }

    // 5. Present the swap chain image
    const vk::PresentInfoKHR presentInfoKHR {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphore,
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };
    auto presentResult = queue.presentKHR(presentInfoKHR);

    switch (presentResult) {
      case vk::Result::eSuccess:
        break;
      case vk::Result::eSuboptimalKHR:
        std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
        break;
      default:
        break; // an unexpected result is returned!
    }
  }

  void createSyncObjects() {
    presentCompleteSemaphore = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());
    renderFinishedSemaphore = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());
    drawFence = vk::raii::Fence(device, {.flags = vk::FenceCreateFlagBits::eSignaled});
  }
};

int main() {
  HelloTriangleApplication app;

  std::cout << glm::asin(1.0) << std::endl;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}