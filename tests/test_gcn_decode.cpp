// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the GCN decoder on bytes that are not code (a stale or reused region behind a
// valid-looking shader header, read from the game's memory): a TranslationError the pipeline
// cache turns into a skipped draw, not an assertion that stops the port.
#include <cassert>
#include <cstdio>
#include <string_view>
#include <vector>
#include "shader_recompiler/frontend/decode.h"

using namespace Shader::Gcn;

int main() {
    {
        // s_endpgm (SOPP): a valid word decodes as before.
        std::vector<u32> code{0xBF810000u, 0xBF810000u};
        GcnCodeSlice slice(code.data(), code.data() + code.size());
        GcnDecodeContext context;
        const GcnInst inst = context.decodeInstruction(slice);
        assert(inst.opcode == Opcode::S_ENDPGM);
    }
    {
        // The word from a crash log (upstream #144): no GCN encoding starts like this.
        std::vector<u32> code{0xe7984924u, 0u, 0u, 0u};
        GcnCodeSlice slice(code.data(), code.data() + code.size());
        GcnDecodeContext context;
        bool thrown = false;
        try {
            (void)context.decodeInstruction(slice);
        } catch (const Shader::TranslationError& error) {
            thrown = true;
            assert(std::string_view(error.what()).find("illegal encoding") != std::string_view::npos);
        }
        assert(thrown);
    }
    std::puts("PASS: GCN decoder: valid word decodes, garbage throws TranslationError");
    return 0;
}
