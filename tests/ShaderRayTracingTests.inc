void TestRayTracingDispatchDetection() {
  namespace D = ShaderRecompiler::Decoder;
  const auto check_decoded = [](std::span<const uint32_t> code, bool expected,
                                const char *message) {
    D::Program decoded;
    D::DecodeProgram(code, decoded);
    Check(decoded.has_bvh == expected, message);
    if (expected) {
      const auto &last = decoded.instructions.back();
      Check(last.family == D::Family::MIMG &&
                (last.opcode_id == 0xe6u || last.opcode_id == 0xe7u),
            "decoder continued past the first BVH instruction");
      const auto result = ShaderRecompiler::TranslateProgram(
          code, MakeCompileOptions(ShaderType::Compute));
      Check(result.skip_dispatch && result.program.blocks.empty() && result.cfg_dump.empty(),
            "BVH compute shader reached IR translation");
    }
  };
  const auto check = [&](std::initializer_list<uint32_t> code, bool expected,
                         const char *message) {
    check_decoded(std::span<const uint32_t>{code.begin(), code.size()}, expected, message);
  };

  check({0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "captured contiguous BVH instruction did not skip dispatch");
  check({0xf1989f07u, 0x00071001u, 0x3d3e3c15u, 0x38095851u, 0x00003a39u,
         EncodeSopp(0x01)}, true,
        "captured NSA BVH instruction did not skip dispatch");
  check({0xf19c9f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "RDNA2 BVH64 instruction did not skip dispatch");
  check({0xbe8e210eu, 0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "unsupported SWAPPC before BVH prevented dispatch detection");
  check({0xf1989f01u, 0x00081014u, 0xc0000000u, EncodeSopp(0x01)}, true,
        "unknown instruction after BVH prevented dispatch skipping");

  for (uint32_t opcode : {0xe6u, 0xe7u}) {
    for (uint32_t nsa = 0; nsa < 4; ++nsa) {
      for (uint32_t slc = 0; slc < 2; ++slc) {
        std::array<uint32_t, 6> code{
            EncodeMimg0(opcode, 0xf, true) | (nsa << 1u) | (slc << 25u),
            EncodeMimg1(0, 0, 0, 0)};
        code[2 + nsa] = EncodeSopp(0x01);
        check_decoded(std::span(code).first(3 + nsa), true,
                      "BVH cache flags or NSA length prevented dispatch detection");
      }
    }
  }
  for (uint32_t opcode : {0x66u, 0x67u, 0xe4u, 0xe5u, 0xe8u, 0xe9u}) {
    check({EncodeMimg0(opcode, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
          false, "non-BVH image opcode was incorrectly skipped");
  }

  check({EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "scalar literal was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf) | 6u, EncodeMimg1(0, 0, 0, 0),
         0xf1989f01u, 0xf19c9f01u, 0u, EncodeSopp(0x01)}, false,
        "NSA address words were mistaken for BVH instructions");
  check({EncodeMimg0(0x20, 0xf), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "MIMG operand word was mistaken for a BVH instruction");
  check({EncodeSopp(0x01), 0xf1989f01u, 0x00081014u}, false,
        "data after the shader end was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
        false, "ordinary image shader was incorrectly skipped");

  // An end instruction on one branch does not end another branch's code.
  check({EncodeSopp(0x04, 1), EncodeSopp(0x01), 0xf1989f01u, 0x00081014u,
         EncodeSopp(0x01)}, true,
        "BVH at a branch target after an early end was not detected");

  const std::array bvh_code{0xf1989f01u, 0x00081014u, 0xf19c9f01u, 0x00081014u,
                            EncodeSopp(0x01)};
  const std::array ordinary_code{EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)};
  D::Program reused;
  D::DecodeProgram(bvh_code, reused);
  Check(reused.has_bvh && reused.instructions.size() == 1,
        "decoder did not stop at the first BVH instruction");
  D::DecodeProgram(ordinary_code, reused);
  Check(!reused.has_bvh, "decoder retained stale BVH metadata");
  for (const auto stage : {ShaderType::Compute, ShaderType::Vertex, ShaderType::Pixel}) {
    const auto options = MakeCompileOptions(stage);
    Check(!ShaderRecompiler::TranslateProgram(ordinary_code, options).skip_dispatch,
          "BVH-shaped literal marked an ordinary shader for skipping");
    const auto compiled = RecompileForTest(ordinary_code, options);
    Check(!compiled.spirv.empty(), "ordinary shader did not compile");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
}

void TestRayTracingEmulation() {
  constexpr uint32_t R128 = 1u << 15u;
  constexpr uint32_t Dim1D = 0u;
  struct Variant {
    const char *name;
    uint32_t opcode;
    bool a16;
    std::vector<uint32_t> nsa;
  };
  const std::vector<Variant> variants = {
      {"contiguous BVH", 0xe6u, false, {}},
      // Node pointer in v0, the remaining ten address components in v50..v59.
      {"NSA BVH", 0xe6u, false, {0x35343332u, 0x39383736u, 0x00003b3au}},
      {"contiguous BVH64", 0xe7u, false, {}},
      {"A16 BVH", 0xe6u, true, {}},
  };
  for (const auto &variant : variants) {
    std::vector<uint32_t> shader = {
        // T# in s8..s11: base 0x10000, box sorting, NUM_RECORDS, type 8 + barycentrics.
        EncodeSMovB32(8, 255), 0x00000100u,
        EncodeSMovB32(9, 255), 0x80000000u,
        EncodeSMovB32(10, 255), 0xffffffffu,
        EncodeSMovB32(11, 255), 0x81000000u,
        EncodeMimg0(variant.opcode, 0xf, false, Dim1D) | R128 |
            (static_cast<uint32_t>(variant.nsa.size()) << 1u),
        EncodeMimg1(20, 2, 0, 0, variant.a16),
    };
    shader.insert(shader.end(), variant.nsa.begin(), variant.nsa.end());
    // Store the first result dword so the intersection is not dead code.
    shader.push_back(EncodeMubuf0(0x1c, 0, false));
    shader.push_back(EncodeMubuf1(20, 12, 0));
    shader.push_back(EncodeSopp(0x01));

    auto options = MakeCompileOptions(ShaderType::Compute);
    options.dump_ir = true;
    options.emulate_ray_tracing = true;
    const auto translated = ShaderRecompiler::TranslateProgram(shader, options);
    Check(!translated.skip_dispatch, variant.name);
    const auto result = RecompileForTest(shader, options);
    Check(!result.spirv.empty(), variant.name);
    Check(result.program.info.uses_dma,
          "BVH emulation did not read nodes through guest addresses");
    Check(result.ir_dump.find("LoadAddressU32") != std::string::npos,
          "BVH emulation did not emit node loads");
    CheckSpirvBinaryValidates(result.spirv);
  }
}
