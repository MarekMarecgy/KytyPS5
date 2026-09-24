#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <algorithm>
#include <array>

// Software emulation of IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY (RDNA2 ISA 8.2.10).
//
// The instruction tests one BVH node against a ray; the traversal loop itself is guest code.
// Node memory layout follows the RDNA2 hardware format (as used by AMD GPURT and Mesa RADV):
//   node pointer: bits [2:0] node type, remaining bits address the node in 64-byte units
//                 relative to the T# base address (T# dwords 0-1 hold base >> 8)
//   types 0-3:    triangle node, 5 vertices (dwords 0-14); the type selects the triangle
//   type 4:       box16 node, 4 child pointers + 4 AABBs packed as fp16 (dwords 4-15)
//   type 5:       box32 node, 4 child pointers + 4 AABBs as fp32 (dwords 4-27)
// Box results are the child pointers sorted by entry distance (0xffffffff for misses).
// Triangle results are {t_num, t_denom, triangle_id, hit_status} or, with T# bit 120 set,
// {t_num, t_denom, i_num, j_num}; a miss returns t_num = +inf.
//
// The emulation is branch-free: every lane evaluates both paths with predicated loads, so a
// malformed node can produce wrong results but never an unbounded loop.

namespace Libs::Graphics::ShaderRecompiler::Frontend {

namespace {

constexpr uint32_t InvalidNode = 0xffffffffu;
constexpr uint32_t PosInfBits  = 0x7f800000u;

struct Vec3 {
	IR::F32 x;
	IR::F32 y;
	IR::F32 z;
};

} // namespace

bool Translator::IMAGE_BVH_INTERSECT_RAY(const Decoder::Instruction& inst) {
	const bool bvh64 = inst.opcode == Decoder::Opcode::IMAGE_BVH64_INTERSECT_RAY;
	const bool a16   = (inst.image_sample_flags & Decoder::ImageSampleFlagA16) != 0u;
	if (inst.src1.kind != Decoder::OperandKind::Sgpr) {
		return false;
	}

	const auto u32 = [](uint32_t value) { return IR::U32(IR::Value(value)); };
	const auto f32 = [](float value) { return IR::F32(IR::Value::F32(value)); };
	const auto fop = [&](IR::ValueOpcode opcode, IR::F32 lhs, IR::F32 rhs) {
		return IR::F32(ir.Emit(opcode, {lhs, rhs}));
	};
	const auto add = [&](IR::F32 lhs, IR::F32 rhs) {
		return fop(IR::ValueOpcode::FPAdd32, lhs, rhs);
	};
	const auto sub = [&](IR::F32 lhs, IR::F32 rhs) {
		return fop(IR::ValueOpcode::FPSub32, lhs, rhs);
	};
	const auto mul = [&](IR::F32 lhs, IR::F32 rhs) {
		return fop(IR::ValueOpcode::FPMul32, lhs, rhs);
	};
	const auto fmin = [&](IR::F32 lhs, IR::F32 rhs) {
		return fop(IR::ValueOpcode::FPMin32, lhs, rhs);
	};
	const auto fmax = [&](IR::F32 lhs, IR::F32 rhs) {
		return fop(IR::ValueOpcode::FPMax32, lhs, rhs);
	};
	const auto fcmp = [&](IR::ValueOpcode opcode, IR::F32 lhs, IR::F32 rhs) {
		return IR::U1(ir.Emit(opcode, {lhs, rhs}));
	};
	const auto fge = [&](IR::F32 lhs, IR::F32 rhs) {
		return fcmp(IR::ValueOpcode::FPOrdGreaterThanEqual32, lhs, rhs);
	};
	const auto fle = [&](IR::F32 lhs, IR::F32 rhs) {
		return fcmp(IR::ValueOpcode::FPOrdLessThanEqual32, lhs, rhs);
	};
	const auto all = [&](std::initializer_list<IR::U1> values) {
		IR::U1 result = *values.begin();
		for (auto it = values.begin() + 1; it != values.end(); ++it) {
			result = ir.LogicalAnd(result, *it);
		}
		return result;
	};
	const auto half = [&](IR::U32 word, bool high) {
		const IR::U32 bits = high ? ir.ShiftRightLogical(word, u32(16u)) : word;
		const auto    u16  = ir.Emit(IR::ValueOpcode::ConvertU16U32, {bits});
		const auto    f16  = ir.Emit(IR::ValueOpcode::BitCastF16U16, {u16});
		return IR::F32(ir.Emit(IR::ValueOpcode::ConvertF32F16, {f16}));
	};
	const auto select3 = [&](IR::U1 condition, const Vec3& lhs, const Vec3& rhs) {
		return Vec3 {SelectF32(condition, lhs.x, rhs.x), SelectF32(condition, lhs.y, rhs.y),
		             SelectF32(condition, lhs.z, rhs.z)};
	};
	const auto sub3 = [&](const Vec3& lhs, const Vec3& rhs) {
		return Vec3 {sub(lhs.x, rhs.x), sub(lhs.y, rhs.y), sub(lhs.z, rhs.z)};
	};
	const auto dot3 = [&](const Vec3& lhs, const Vec3& rhs) {
		return add(add(mul(lhs.x, rhs.x), mul(lhs.y, rhs.y)), mul(lhs.z, rhs.z));
	};
	const auto cross3 = [&](const Vec3& lhs, const Vec3& rhs) {
		return Vec3 {sub(mul(lhs.y, rhs.z), mul(lhs.z, rhs.y)),
		             sub(mul(lhs.z, rhs.x), mul(lhs.x, rhs.z)),
		             sub(mul(lhs.x, rhs.y), mul(lhs.y, rhs.x))};
	};

	// Address VGPRs, honoring the non-sequential-address (NSA) encoding.
	const auto nsa_components =
	    std::min(inst.image_nsa_dwords * 4u, Decoder::MaxImageNsaAddressComponents);
	const auto address = [&](uint32_t index) -> IR::U32 {
		if (index != 0u && index - 1u < nsa_components) {
			return ir.GetVectorReg(static_cast<IR::VectorReg>(inst.image_nsa_addr[index - 1u]));
		}
		return ReadRawU32(OffsetOperand(PlainOperand(inst.src0), index));
	};
	const auto address_f32 = [&](uint32_t index) { return ir.BitCastF32(address(index)); };

	uint32_t   cursor  = 0;
	const auto node_lo = address(cursor++);
	const auto node_hi = bvh64 ? address(cursor++) : u32(0u);
	const auto extent  = address_f32(cursor++);
	const Vec3 origin  = {address_f32(cursor), address_f32(cursor + 1u), address_f32(cursor + 2u)};
	cursor += 3u;
	Vec3 dir;
	Vec3 inv_dir;
	if (a16) {
		const auto w0 = address(cursor);
		const auto w1 = address(cursor + 1u);
		const auto w2 = address(cursor + 2u);
		dir           = {half(w0, false), half(w0, true), half(w1, false)};
		inv_dir       = {half(w1, true), half(w2, false), half(w2, true)};
	} else {
		dir     = {address_f32(cursor), address_f32(cursor + 1u), address_f32(cursor + 2u)};
		inv_dir = {address_f32(cursor + 3u), address_f32(cursor + 4u), address_f32(cursor + 5u)};
	}

	// T#: base address (>> 8) in bits [39:0], box sorting in bit 63, triangle return mode in 120.
	const auto resource_index = inst.src1.reg / 4u;
	const auto desc0          = GetResourceDword(resource_index, 0);
	const auto desc1          = GetResourceDword(resource_index, 1);
	const auto desc3          = GetResourceDword(resource_index, 3);
	const auto box_sort       = ir.INotEqual(ir.BitwiseAnd(desc1, u32(0x80000000u)), u32(0u));
	const auto barycentrics   = ir.INotEqual(ir.BitwiseAnd(desc3, u32(1u << 24u)), u32(0u));

	const auto base =
	    IR::U64(ir.Emit(IR::ValueOpcode::ShiftLeftLogical64,
	                    {ir.ConstructU64(desc0, ir.BitwiseAnd(desc1, u32(0xffu))), u32(8u)}));
	const auto node_offset =
	    IR::U64(ir.Emit(IR::ValueOpcode::ShiftLeftLogical64,
	                    {ir.Emit(IR::ValueOpcode::BitwiseAnd64,
	                             {ir.ConstructU64(node_lo, node_hi), IR::Value(~uint64_t {7u})}),
	                     u32(3u)}));
	const auto node_address = ir.Emit(IR::ValueOpcode::IAdd64, {base, node_offset});
	const auto address_lo =
	    IR::U32(ir.Emit(IR::ValueOpcode::CompositeExtractU64, {node_address, IR::Value(0u)}));
	const auto address_hi =
	    IR::U32(ir.Emit(IR::ValueOpcode::CompositeExtractU64, {node_address, IR::Value(1u)}));

	const auto type     = ir.BitwiseAnd(node_lo, u32(7u));
	const auto valid    = bvh64 ? ir.LogicalNot(ir.LogicalAnd(ir.IEqual(node_lo, u32(InvalidNode)),
	                                                          ir.IEqual(node_hi, u32(InvalidNode))))
	                            : ir.INotEqual(node_lo, u32(InvalidNode));
	const auto is_tri   = ir.ULessThan(type, u32(4u));
	const auto is_box16 = ir.IEqual(type, u32(4u));
	const auto is_box32 = ir.LogicalAnd(ir.LogicalNot(is_tri), ir.LogicalNot(is_box16));

	const auto node_resource = GetAddressResource(address_lo, address_hi);
	const auto load          = [&](uint32_t dword, IR::U1 active) {
		IR::MemoryInfo memory;
		memory.kind            = IR::ResourceKind::Global;
		memory.address_is_full = true;
		memory.offset          = dword * 4u;
		memory.data_bits       = 32u;
		memory.data_dwords     = 1u;
		memory.component_count = 1u;
		return IR::U32(ir.Emit(IR::ValueOpcode::LoadAddressU32,
		                       {node_resource, address_lo, address_hi, active},
		                       AddMemoryInfo(memory, inst.pc)));
	};
	const auto              active       = ir.LogicalAnd(ir.GetExec(), valid);
	const auto              active_box32 = ir.LogicalAnd(active, is_box32);
	std::array<IR::U32, 28> dw {};
	for (uint32_t i = 0; i < 16u; i++) {
		dw[i] = load(i, active);
	}
	for (uint32_t i = 16; i < 28u; i++) {
		dw[i] = load(i, active_box32);
	}

	// Box node: 4 slab tests, then a 5-comparator sorting network on entry distance.
	const auto             zero = f32(0.0f);
	const auto             inf  = ir.BitCastF32(u32(PosInfBits));
	std::array<IR::F32, 4> dist {};
	std::array<IR::U32, 4> child {};
	for (uint32_t i = 0; i < 4u; i++) {
		const auto b32   = 4u + i * 6u;
		const Vec3 min32 = {ir.BitCastF32(dw[b32]), ir.BitCastF32(dw[b32 + 1u]),
		                    ir.BitCastF32(dw[b32 + 2u])};
		const Vec3 max32 = {ir.BitCastF32(dw[b32 + 3u]), ir.BitCastF32(dw[b32 + 4u]),
		                    ir.BitCastF32(dw[b32 + 5u])};
		const auto b16   = 4u + i * 3u;
		const Vec3 min16 = {half(dw[b16], false), half(dw[b16], true), half(dw[b16 + 1u], false)};
		const Vec3 max16 = {half(dw[b16 + 1u], true), half(dw[b16 + 2u], false),
		                    half(dw[b16 + 2u], true)};
		const auto lo    = select3(is_box16, min16, min32);
		const auto hi    = select3(is_box16, max16, max32);

		const auto t0x  = mul(sub(lo.x, origin.x), inv_dir.x);
		const auto t0y  = mul(sub(lo.y, origin.y), inv_dir.y);
		const auto t0z  = mul(sub(lo.z, origin.z), inv_dir.z);
		const auto t1x  = mul(sub(hi.x, origin.x), inv_dir.x);
		const auto t1y  = mul(sub(hi.y, origin.y), inv_dir.y);
		const auto t1z  = mul(sub(hi.z, origin.z), inv_dir.z);
		const auto tmin = fmax(fmax(fmin(t0x, t1x), fmin(t0y, t1y)), fmin(t0z, t1z));
		const auto tmax = fmin(fmin(fmax(t0x, t1x), fmax(t0y, t1y)), fmax(t0z, t1z));

		// A NaN min.x marks an inactive child.
		const auto hit = all({valid, ir.INotEqual(dw[i], u32(InvalidNode)),
		                      ir.LogicalNot(IR::U1(ir.Emit(IR::ValueOpcode::FPIsNan32, {lo.x}))),
		                      fge(tmax, fmax(tmin, zero)), fle(tmin, extent)});
		dist[i]        = SelectF32(hit, tmin, inf);
		child[i]       = ir.Select(hit, dw[i], u32(InvalidNode));
	}
	const auto compare_swap = [&](uint32_t a, uint32_t b) {
		const auto swap =
		    ir.LogicalAnd(box_sort, fcmp(IR::ValueOpcode::FPOrdLessThan32, dist[b], dist[a]));
		const auto dist_a  = SelectF32(swap, dist[b], dist[a]);
		const auto dist_b  = SelectF32(swap, dist[a], dist[b]);
		const auto child_a = ir.Select(swap, child[b], child[a]);
		const auto child_b = ir.Select(swap, child[a], child[b]);
		dist[a]            = dist_a;
		dist[b]            = dist_b;
		child[a]           = child_a;
		child[b]           = child_b;
	};
	compare_swap(0, 1);
	compare_swap(2, 3);
	compare_swap(0, 2);
	compare_swap(1, 3);
	compare_swap(1, 2);

	// Triangle node: the node type selects one of four triangles over five shared vertices.
	std::array<Vec3, 5> v {};
	for (uint32_t k = 0; k < 5u; k++) {
		v[k] = {ir.BitCastF32(dw[k * 3u]), ir.BitCastF32(dw[k * 3u + 1u]),
		        ir.BitCastF32(dw[k * 3u + 2u])};
	}
	const auto is_type = [&](uint32_t value) { return ir.IEqual(type, u32(value)); };
	const auto pick    = [&](const Vec3& t0, const Vec3& t1, const Vec3& t2, const Vec3& t3) {
		return select3(is_type(0), t0, select3(is_type(1), t1, select3(is_type(2), t2, t3)));
	};
	const auto a = pick(v[0], v[1], v[2], v[2]);
	const auto b = pick(v[1], v[3], v[3], v[4]);
	const auto c = pick(v[2], v[2], v[4], v[0]);

	// Moller-Trumbore with the division deferred: t = t_num / det, i = i_num / det, j = j_num /
	// det.
	const auto e1       = sub3(b, a);
	const auto e2       = sub3(c, a);
	const auto p        = cross3(dir, e2);
	const auto det      = dot3(e1, p);
	const auto s        = sub3(origin, a);
	const auto i_num    = dot3(s, p);
	const auto q        = cross3(s, e1);
	const auto j_num    = dot3(dir, q);
	const auto t_num    = dot3(e2, q);
	const auto negative = fcmp(IR::ValueOpcode::FPOrdLessThan32, det, zero);
	const auto sign     = SelectF32(negative, f32(-1.0f), f32(1.0f));
	const auto abs_det  = mul(det, sign);
	const auto i_signed = mul(i_num, sign);
	const auto j_signed = mul(j_num, sign);
	const auto t_signed = mul(t_num, sign);
	const auto tri_hit =
	    all({valid, fcmp(IR::ValueOpcode::FPOrdNotEqual32, det, zero), fge(i_signed, zero),
	         fge(j_signed, zero), fle(add(i_signed, j_signed), abs_det), fge(t_signed, zero),
	         fle(t_signed, mul(extent, abs_det))});

	const auto                   triangle_id = dw[15];
	const std::array<IR::U32, 4> tri         = {
	    ir.Select(tri_hit, ir.BitCastU32(t_num), u32(PosInfBits)),
	    ir.Select(tri_hit, ir.BitCastU32(det), ir.BitCastU32(f32(1.0f))),
	    ir.Select(tri_hit, ir.Select(barycentrics, ir.BitCastU32(i_num), triangle_id), u32(0u)),
	    ir.Select(tri_hit, ir.Select(barycentrics, ir.BitCastU32(j_num), u32(1u)), u32(0u)),
	};

	for (uint32_t k = 0; k < 4u; k++) {
		WriteOperand(OffsetOperand(inst.dst, k), ir.Select(is_tri, tri[k], child[k]));
	}
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
