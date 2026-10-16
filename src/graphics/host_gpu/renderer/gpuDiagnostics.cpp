#include "graphics/host_gpu/renderer/gpuDiagnostics.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

struct CheckpointEntry {
	uint64_t          id           = 0;
	uint64_t          tick         = 0;
	uint64_t          shader_hash  = 0;
	uint64_t          shader_hash2 = 0;
	uint64_t          address      = 0;
	uint32_t          x            = 0;
	uint32_t          y            = 0;
	uint32_t          z            = 0;
	GpuCheckpointKind kind         = GpuCheckpointKind::DispatchBegin;
};

constexpr uint64_t RingSize = 1u << 16u;

std::array<CheckpointEntry, RingSize> g_ring {};
std::atomic<uint64_t>                 g_next_id {1};
std::atomic<bool>                     g_reported {false};

const char* KindName(GpuCheckpointKind kind) {
	switch (kind) {
		case GpuCheckpointKind::DispatchBegin: return "dispatch-begin";
		case GpuCheckpointKind::DispatchEnd: return "dispatch-end";
		case GpuCheckpointKind::DispatchIndirectBegin: return "dispatch-indirect-begin";
		case GpuCheckpointKind::DispatchIndirectEnd: return "dispatch-indirect-end";
		case GpuCheckpointKind::DrawBegin: return "draw-begin";
		case GpuCheckpointKind::DrawEnd: return "draw-end";
	}
	return "?";
}

void Print(const char* format, ...) {
	char    line[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(line, sizeof(line), format, args);
	va_end(args);
	LOGF("%s", line);
	std::printf("%s", line);
}

void PrintEntry(const char* prefix, uint64_t id) {
	const auto& entry = g_ring[id & (RingSize - 1u)];
	if (entry.id != id) {
		Print("%s #%" PRIu64 " (overwritten)\n", prefix, id);
		return;
	}
	Print("%s #%" PRIu64 " %s tick=%" PRIu64 " hash=0x%016" PRIx64 " hash2=0x%016" PRIx64
	      " addr=0x%012" PRIx64 " groups=%ux%ux%u\n",
	      prefix, id, KindName(entry.kind), entry.tick, entry.shader_hash, entry.shader_hash2,
	      entry.address, entry.x, entry.y, entry.z);
}

void ReportCheckpoints(const GraphicContext& graphics) {
	uint32_t count = 0;
	graphics.queue.getCheckpointDataNV(&count, nullptr);
	std::vector<vk::CheckpointDataNV> data(count);
	graphics.queue.getCheckpointDataNV(&count, data.data());
	data.resize(count);
	Print("GPU checkpoints: %u reported by the queue\n", count);
	uint64_t newest = 0;
	for (const auto& checkpoint: data) {
		const auto id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(checkpoint.pCheckpointMarker));
		newest        = std::max(newest, id);
		const auto prefix = "  stage " + vk::to_string(checkpoint.stage) + ":";
		PrintEntry(prefix.c_str(), id);
	}
	// Context: everything recorded around the newest marker the GPU reached.
	const auto recorded = g_next_id.load(std::memory_order_relaxed) - 1u;
	if (newest == 0) {
		newest = recorded;
	}
	const uint64_t first = newest > 16u ? newest - 16u : 1u;
	const uint64_t last  = std::min(recorded, newest + 16u);
	Print("GPU checkpoints recorded #%" PRIu64 "..#%" PRIu64 " (newest recorded #%" PRIu64 "):\n",
	      first, last, recorded);
	for (uint64_t id = first; id <= last; id++) {
		PrintEntry(id == newest ? "  >>" : "    ", id);
	}
}

void ReportDeviceFault(const GraphicContext& graphics) {
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Print("vkGetDeviceFaultInfoEXT(counts) failed: %s\n", vk::to_string(result).c_str());
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendors.data();
	result             = graphics.device.getFaultInfoEXT(&counts, &info);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Print("vkGetDeviceFaultInfoEXT failed: %s\n", vk::to_string(result).c_str());
		return;
	}
	Print("GPU fault: \"%s\" addresses=%u vendor=%u\n", info.description.data(),
	      counts.addressInfoCount, counts.vendorInfoCount);
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& a = addresses[i];
		Print("  address[%u]: %s 0x%016" PRIx64 " precision=0x%" PRIx64 "\n", i,
		      vk::to_string(a.addressType).c_str(), static_cast<uint64_t>(a.reportedAddress),
		      static_cast<uint64_t>(a.addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& v = vendors[i];
		Print("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		      v.description.data(), static_cast<uint64_t>(v.vendorFaultCode),
		      static_cast<uint64_t>(v.vendorFaultData));
	}
}

} // namespace

bool GpuDiagnosticsRequested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_CHECKPOINTS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return requested;
}

void GpuCheckpoint(const GraphicContext& graphics, vk::CommandBuffer buffer,
                   GpuCheckpointKind kind, uint64_t tick, uint64_t shader_hash,
                   uint64_t shader_hash2, uint64_t address, uint32_t x, uint32_t y, uint32_t z) {
	if (!graphics.device_checkpoints_enabled) {
		return;
	}
	const auto id = g_next_id.fetch_add(1, std::memory_order_relaxed);
	auto&      entry = g_ring[id & (RingSize - 1u)];
	entry = {id, tick, shader_hash, shader_hash2, address, x, y, z, kind};
	buffer.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(id)));
}

void GpuReportDeviceLost(const GraphicContext& graphics, const char* what, vk::Result result) {
	if (g_reported.exchange(true)) {
		return;
	}
	Print("=== GPU device lost: %s -> %s ===\n", what, vk::to_string(result).c_str());
	if (graphics.device_fault_enabled) {
		ReportDeviceFault(graphics);
	}
	if (graphics.device_checkpoints_enabled) {
		ReportCheckpoints(graphics);
	}
	if (!graphics.device_fault_enabled && !graphics.device_checkpoints_enabled) {
		Print("(set KYTY_GPU_CHECKPOINTS=1 for checkpoint and fault details)\n");
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics
