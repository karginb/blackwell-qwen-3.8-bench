#!/usr/bin/env python3
"""
Inspect a GGUF file and report how quantization bits are actually distributed.

WHY THIS MATTERS
A filename like "IQ2_XXS" reads as "this model is 2-bit", but that is only an
average label. GGUF stores a quantization type PER TENSOR, and modern quants mix
many of them in a single file: parts of the network that tolerate noise get
crushed hard, parts that don't are protected.

This script parses the GGUF header directly (no llama.cpp dependency) and breaks
the file down two ways:

  * by quantization type  -- which formats are present and how much they weigh
  * by tensor role        -- which parts of the network got the extra bits

Usage:
    python3 gguf_inspect.py model.gguf
"""

import re
import struct
import sys
from collections import defaultdict

# ggml type id -> (name, bits per weight, including block scales/mins)
GGML_TYPES = {
    0: ("F32", 32.0),
    1: ("F16", 16.0),
    2: ("Q4_0", 4.5),
    3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5),
    7: ("Q5_1", 6.0),
    8: ("Q8_0", 8.5),
    9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.5625),
    11: ("Q3_K", 3.4375),
    12: ("Q4_K", 4.5),
    13: ("Q5_K", 5.5),
    14: ("Q6_K", 6.5625),
    15: ("Q8_K", 8.0),
    16: ("IQ2_XXS", 2.0625),
    17: ("IQ2_XS", 2.3125),
    18: ("IQ3_XXS", 3.0625),
    19: ("IQ1_S", 1.5625),
    20: ("IQ4_NL", 4.5),
    21: ("IQ3_S", 3.4375),
    22: ("IQ2_S", 2.5),
    23: ("IQ4_XS", 4.25),
    29: ("IQ1_M", 1.75),
    30: ("BF16", 16.0),
}

# GGUF metadata value type -> struct format character
SCALAR_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I",
    5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}

TYPE_STRING = 8
TYPE_ARRAY = 9


class GGUFReader:
    """Minimal GGUF header parser -- enough to enumerate tensors and metadata."""

    def __init__(self, path):
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file (magic was {magic!r})")
        self.version = struct.unpack("<I", self.f.read(4))[0]
        self.n_tensors = struct.unpack("<Q", self.f.read(8))[0]
        self.n_kv = struct.unpack("<Q", self.f.read(8))[0]

    def _read_string(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _skip_value(self, vtype):
        if vtype == TYPE_STRING:
            self._read_string()
        elif vtype == TYPE_ARRAY:
            elem_type = struct.unpack("<I", self.f.read(4))[0]
            length = struct.unpack("<Q", self.f.read(8))[0]
            if elem_type == TYPE_STRING:
                for _ in range(length):
                    self._read_string()
            else:
                self.f.read(struct.calcsize(SCALAR_FMT[elem_type]) * length)
        else:
            self.f.read(struct.calcsize(SCALAR_FMT[vtype]))

    def read_metadata(self, keep_keys=()):
        """Consume the metadata block, returning values whose key contains a keyword."""
        found = {}
        for _ in range(self.n_kv):
            key = self._read_string()
            vtype = struct.unpack("<I", self.f.read(4))[0]
            if vtype == TYPE_STRING and any(k in key for k in keep_keys):
                found[key] = self._read_string()
            else:
                self._skip_value(vtype)
        return found

    def read_tensors(self):
        """Return [(name, element_count, ggml_type_id)] for every tensor."""
        tensors = []
        for _ in range(self.n_tensors):
            name = self._read_string()
            n_dims = struct.unpack("<I", self.f.read(4))[0]
            dims = struct.unpack(f"<{n_dims}Q", self.f.read(8 * n_dims))
            ttype = struct.unpack("<I", self.f.read(4))[0]
            self.f.read(8)  # tensor data offset, not needed here
            count = 1
            for d in dims:
                count *= d
            tensors.append((name, count, ttype))
        return tensors

    def close(self):
        self.f.close()


def classify(name):
    """Group a tensor by its role in the network."""
    if "token_embd" in name:
        return "token embedding"
    if name.startswith("output") or "output.weight" in name:
        return "output head (lm_head)"
    if "nextn" in name:
        return "MTP / next-token head"
    if "ffn_down" in name:
        return "FFN down"
    if "ffn_gate" in name or "ffn_up" in name:
        return "FFN gate/up"
    if re.search(r"attn_(q|k|v|output)", name):
        return "attention"
    if "ssm" in name or "linear_attn" in name or "conv" in name:
        return "linear attention (SSM)"
    if "norm" in name:
        return "norm"
    return "other"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <model.gguf>", file=sys.stderr)
        return 1

    path = sys.argv[1]
    reader = GGUFReader(path)
    meta = reader.read_metadata(keep_keys=("architecture", "general.name"))
    tensors = reader.read_tensors()
    reader.close()

    by_type = defaultdict(lambda: [0, 0.0])  # type name -> [params, bytes]
    by_role = defaultdict(lambda: [0, 0.0])
    total_params = 0
    total_bytes = 0.0

    for name, count, ttype in tensors:
        type_name, bpw = GGML_TYPES.get(ttype, (f"UNKNOWN_{ttype}", 32.0))
        size = count * bpw / 8

        by_type[type_name][0] += count
        by_type[type_name][1] += size
        role = classify(name)
        by_role[role][0] += count
        by_role[role][1] += size
        total_params += count
        total_bytes += size

    print(f"file         : {path.split('/')[-1]}")
    print(f"architecture : {meta.get('general.architecture', '?')}")
    print(f"model name   : {meta.get('general.name', '?')}")
    print(f"tensors      : {len(tensors)}")
    print(f"parameters   : {total_params / 1e9:.2f} B")
    print(f"size         : {total_bytes / 1e9:.2f} GB")
    print(f"AVERAGE      : {total_bytes * 8 / total_params:.2f} bits per weight")
    print()

    bpw_of = {n: w for n, w in GGML_TYPES.values()}

    print("=== BY QUANTIZATION TYPE ===")
    header = f"{'type':<10}{'bpw':>7}{'params':>13}{'size':>10}{'share':>8}"
    print(header)
    print("-" * len(header))
    for type_name, (params, size) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        print(
            f"{type_name:<10}{bpw_of.get(type_name, 0):>7.2f}{params / 1e9:>12.2f}B"
            f"{size / 1e9:>9.2f}G{size / total_bytes * 100:>7.1f}%"
        )
    print()

    print("=== BY TENSOR ROLE (where the extra bits went) ===")
    header = f"{'role':<26}{'params':>12}{'size':>9}{'avg bpw':>9}"
    print(header)
    print("-" * len(header))
    for role, (params, size) in sorted(by_role.items(), key=lambda kv: -kv[1][1]):
        print(f"{role:<26}{params / 1e9:>10.2f}B{size / 1e9:>8.2f}G{size * 8 / params:>9.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
