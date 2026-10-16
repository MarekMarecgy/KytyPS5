#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAGNOSTICS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAGNOSTICS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Opt-in device-lost diagnostics (KYTY_GPU_CHECKPOINTS=1). Commands are bracketed with
// VK_NV_device_diagnostic_checkpoints markers; after a device loss the last markers the GPU
// reached are printed together with VK_EXT_device_fault information.
enum class GpuCheckpointKind : uint32_t {
	DispatchBegin,
	DispatchEnd,
	DispatchIndirectBegin,
	DispatchIndirectEnd,
	DrawBegin,
	DrawEnd,
};

[[nodiscard]] bool GpuDiagnosticsRequested();

void GpuCheckpoint(const GraphicContext& graphics, vk::CommandBuffer buffer,
                   GpuCheckpointKind kind, uint64_t tick, uint64_t shader_hash,
                   uint64_t shader_hash2, uint64_t address, uint32_t x = 0, uint32_t y = 0,
                   uint32_t z = 0);

// Prints the diagnostics once; later calls do nothing.
void GpuReportDeviceLost(const GraphicContext& graphics, const char* what, vk::Result result);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUDIAGNOSTICS_H_ */
