#!/usr/bin/env python3
"""Generate ml_classifier.h -- a header-only C++ port of the ML interface
classifier -- from a TorchScript model.

Standalone port of generate_fortran_classifier.ipynb that emits C++ instead of
Fortran.  The static scaffolding (banner comment, geometry constants, Stencil
struct, preprocessing, entry points) is reproduced verbatim; the weight/bias
tables and every architecture-dependent line are generated from the model.

Usage:
    python generate_cpp_classifier.py [model.pt] [-o ml_classifier.h]
"""

import argparse

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# Number of values printed per source line inside the weight/bias tables.
VALS_PER_LINE = 6
# Printf-style format for every table entry.
VAL_FMT = "%.8e"


# ---------------------------------------------------------------------------
#  Static scaffolding
# ---------------------------------------------------------------------------
HEADER_TOP = '// ---------------------------------------------------------------------------\n//  ml_classifier.hpp -- header-only C++ port of the torch-free Fortran\n//  ML interface classifier (ml_classifier.f90 + ml_classifier_c_api.f90).\n//\n//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n//  !!!!!!!!!!!!! DO NOT MODIFY -- weights are automatically generated\n//  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n//\n//  Network: {arch}, ReLU on hidden layers, argmax output.\n//\n//  Input layout (identical to the Fortran/C interface it replaces):\n//    vfrac    : 5*5*5   doubles,  vfrac[i + 5*j + 25*k]\n//    liq_bary : 5*5*5*3 doubles,  liq_bary[i + 5*j + 25*k + 125*c]\n//  i.e. Fortran column-major (i fastest), exactly what main.cpp already fills.\n//\n//  Return value (matches ml_classifier_fortran, i.e. Fortran class id - 1):\n//    -1 : no classification (central cell is not mixed)\n//     0 : well-resolved interface\n//     1 : ligament\n//     2 : droplet\n//     3 : sheet/film\n//     4 : ligament end\n//     5 : sheet end\n//\n//  Requires C++17 (inline variables). Single translation unit safe.\n// ---------------------------------------------------------------------------\n#ifndef ML_CLASSIFIER_HPP\n#define ML_CLASSIFIER_HPP\n\n#include <algorithm>\n#include <array>\n#include <cstddef>\n\nnamespace ml_classifier {\n\n// ---- geometry / constants -------------------------------------------------\ninline constexpr int N = 5;\ninline constexpr int CID = N / 2;                 // 0-based centre (Fortran N/2+1)\ninline constexpr int NCELL = N * N * N;           // 125\ninline constexpr int NIN = NCELL * 4;             // 500\ninline constexpr double EPSILON_CONNECT = 1.0e-12;\n\n// 6-connected neighbour offsets\ninline constexpr int N6[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},\n                                 {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};\n\n// Fortran storage order: (i,j,k) with i fastest.\ninline constexpr int idx3(int i, int j, int k) { return i + N * j + N * N * k; }\n\nnamespace detail {\n'

BODY_TOP = "\n}  // namespace detail\n\n// ---------------------------------------------------------------------------\n//  Stencil: a mutable working copy of the 5^3 volume-fraction / barycentre data\n// ---------------------------------------------------------------------------\nstruct Stencil {\n  std::array<double, NCELL> vfrac{};\n  std::array<double, NCELL * 3> bary{};   // component c at bary[cell + NCELL*c]\n\n  double& f(int i, int j, int k) { return vfrac[idx3(i, j, k)]; }\n  double f(int i, int j, int k) const { return vfrac[idx3(i, j, k)]; }\n  double& b(int i, int j, int k, int c) { return bary[idx3(i, j, k) + NCELL * c]; }\n  double b(int i, int j, int k, int c) const { return bary[idx3(i, j, k) + NCELL * c]; }\n};\n\nnamespace detail {\n\n// Mirror the stencil about axis `dir` (0=x, 1=y, 2=z), negating that component.\ninline void reflect(Stencil& s, int dir) {\n  const Stencil t = s;\n  for (int i = 0; i < N; ++i)\n    for (int j = 0; j < N; ++j)\n      for (int k = 0; k < N; ++k) {\n        const int mi = (dir == 0) ? N - 1 - i : i;\n        const int mj = (dir == 1) ? N - 1 - j : j;\n        const int mk = (dir == 2) ? N - 1 - k : k;\n        s.f(i, j, k) = t.f(mi, mj, mk);\n        for (int c = 0; c < 3; ++c)\n          s.b(i, j, k, c) = (dir == c) ? -t.b(mi, mj, mk, c) : t.b(mi, mj, mk, c);\n      }\n}\n\n// Cyclic permutation of the axes: dir==1 -> (x,y,z) from (k,i,j); dir==2 -> (j,k,i).\ninline void permute(Stencil& s, int dir) {\n  if (dir != 1 && dir != 2) return;\n  const Stencil t = s;\n  for (int i = 0; i < N; ++i)\n    for (int j = 0; j < N; ++j)\n      for (int k = 0; k < N; ++k) {\n        if (dir == 1) {\n          s.f(i, j, k) = t.f(k, i, j);\n          s.b(i, j, k, 0) = t.b(k, i, j, 1);\n          s.b(i, j, k, 1) = t.b(k, i, j, 2);\n          s.b(i, j, k, 2) = t.b(k, i, j, 0);\n        } else {\n          s.f(i, j, k) = t.f(j, k, i);\n          s.b(i, j, k, 0) = t.b(j, k, i, 2);\n          s.b(i, j, k, 1) = t.b(j, k, i, 0);\n          s.b(i, j, k, 2) = t.b(j, k, i, 1);\n        }\n      }\n}\n\n// Transpose the x and y axes.\ninline void swap_xy(Stencil& s) {\n  const Stencil t = s;\n  for (int i = 0; i < N; ++i)\n    for (int j = 0; j < N; ++j)\n      for (int k = 0; k < N; ++k) {\n        s.f(i, j, k) = t.f(j, i, k);\n        s.b(i, j, k, 0) = t.b(j, i, k, 1);\n        s.b(i, j, k, 1) = t.b(j, i, k, 0);\n        s.b(i, j, k, 2) = t.b(j, i, k, 2);\n      }\n}\n\n// Index of the largest of three values, ties resolved towards the first.\ninline int largest_value_index(double v0, double v1, double v2) {\n  double largest = v0;\n  int largest_idx = 0;\n  if (v1 > largest) { largest = v1; largest_idx = 1; }\n  if (v2 > largest) { largest_idx = 2; }\n  return largest_idx;\n}\n\n// out[j] = sum_i in[i] * w[j][i] + b[j], optionally ReLU'd.\ntemplate <int In, int Out>\ninline void dense(const double (&x)[In], const double (&w)[Out][In],\n                  const double (&bias)[Out], double (&y)[Out], bool relu) {\n  for (int j = 0; j < Out; ++j) {\n    double acc = bias[j];\n    const double* wj = w[j];\n    for (int i = 0; i < In; ++i) acc += x[i] * wj[i];\n    y[j] = relu ? std::max(0.0, acc) : acc;\n  }\n}\n\n// Flood-fill from the centre, drop disconnected liquid, canonicalise the\n// orientation, then flatten to the 500-element network input.\ninline void preprocess_and_flatten(Stencil& s, double (&flat)[NIN]) {\n  // ---- flood fill (BFS) from the central cell -----------------------------\n  bool visited[NCELL] = {};\n  int q[NCELL][3];\n  int qsize = 0;\n  visited[idx3(CID, CID, CID)] = true;\n  q[qsize][0] = CID; q[qsize][1] = CID; q[qsize][2] = CID;\n  ++qsize;\n\n  for (int qi = 0; qi < qsize; ++qi) {\n    const int i = q[qi][0], j = q[qi][1], k = q[qi][2];\n    for (int m = 0; m < 6; ++m) {\n      const int ni = i + N6[m][0], nj = j + N6[m][1], nk = k + N6[m][2];\n      if (ni < 0 || ni >= N || nj < 0 || nj >= N || nk < 0 || nk >= N) continue;\n      if (visited[idx3(ni, nj, nk)]) continue;\n      if (s.f(ni, nj, nk) > EPSILON_CONNECT) {\n        visited[idx3(ni, nj, nk)] = true;\n        q[qsize][0] = ni; q[qsize][1] = nj; q[qsize][2] = nk;\n        ++qsize;\n      }\n    }\n  }\n\n  // ---- zero everything not connected to the centre ------------------------\n  for (int c = 0; c < NCELL; ++c) {\n    if (visited[c]) continue;\n    if (s.vfrac[c] > 0.0) {\n      s.vfrac[c] = 0.0;\n      s.bary[c] = s.bary[c + NCELL] = s.bary[c + 2 * NCELL] = 0.0;\n    }\n  }\n\n  // ---- global liquid centroid of the stencil ------------------------------\n  double g[3] = {0.0, 0.0, 0.0};\n  double total_volume = 0.0;\n  for (int c = 0; c < NCELL; ++c) {\n    total_volume += s.vfrac[c];\n    g[0] += s.bary[c];\n    g[1] += s.bary[c + NCELL];\n    g[2] += s.bary[c + 2 * NCELL];\n  }\n  if (total_volume != 0.0)\n    for (double& gi : g) gi /= total_volume;\n\n  // ---- reflect into the first octant: cx, cy, cz >= 0 ---------------------\n  for (int d = 0; d < 3; ++d)\n    if (g[d] < 0.0) { reflect(s, d); g[d] = -g[d]; }\n\n  // ---- permute so the dominant axis is x ----------------------------------\n  const int largest_idx = largest_value_index(g[2], g[0], g[1]);\n  if (largest_idx == 1) {\n    permute(s, 1);                       // (cx,cy,cz) -> (cy,cz,cx)\n    const double tmp = g[0];\n    g[0] = g[1];\n    g[1] = g[2];\n    g[2] = tmp;\n  } else if (largest_idx == 2) {\n    permute(s, 2);                       // (cx,cy,cz) -> (cz,cx,cy)\n    const double tmp = g[0];\n    g[0] = g[2];\n    g[2] = g[1];\n    g[1] = tmp;\n  }\n\n  // ---- order the two remaining axes ---------------------------------------\n  if (g[1] > g[0]) {\n    swap_xy(s);\n    std::swap(g[0], g[1]);\n  }\n\n  // ---- flatten: [vfrac, mx, my, mz] per cell, i outer / k inner -----------\n  int pos = 0;\n  for (int i = 0; i < N; ++i)\n    for (int j = 0; j < N; ++j)\n      for (int k = 0; k < N; ++k) {\n        flat[pos++] = s.f(i, j, k);\n        flat[pos++] = s.b(i, j, k, 0);\n        flat[pos++] = s.b(i, j, k, 1);\n        flat[pos++] = s.b(i, j, k, 2);\n      }\n}\n\n}  // namespace detail\n\n// ---------------------------------------------------------------------------\n//  Forward pass. Returns the Fortran class id (0 = unclassified, 1..6 = class).\n//  `s` is modified in place, exactly as the Fortran routine does.\n// ---------------------------------------------------------------------------\n"

BODY_BOTTOM = '\n// ---------------------------------------------------------------------------\n//  Convenience entry point taking raw flat arrays.\n//    vfrac    : NCELL     doubles\n//    liq_bary : NCELL * 3 doubles\n//  Inputs are not modified (an internal copy is used).\n// ---------------------------------------------------------------------------\ninline int classify(const double* vfrac, const double* liq_bary) {\n  Stencil s;\n  std::copy(vfrac, vfrac + NCELL, s.vfrac.begin());\n  std::copy(liq_bary, liq_bary + NCELL * 3, s.bary.begin());\n  return get_class(s) - 1;   // shift to 0-based ids (-1 = unclassified)\n}\n\n}  // namespace ml_classifier\n\n// ---------------------------------------------------------------------------\n//  Drop-in replacement for the Fortran-bound C entry point, so existing call\n//  sites (e.g. main.cpp) keep working unchanged.\n// ---------------------------------------------------------------------------\ninline int ml_classifier_fortran(const double* vfrac, const double* liq_bary) {\n  return ml_classifier::classify(vfrac, liq_bary);\n}\n'

FOOTER = "#endif  // ML_CLASSIFIER_HPP"


# ---------------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------------
def format_values(values, out):
    """Append `values` as comma-separated literals, VALS_PER_LINE per line.

    No trailing comma follows the final value; the caller adds whatever closing
    punctuation the surrounding brace needs.
    """
    n = len(values)
    for start in range(0, n, VALS_PER_LINE):
        chunk = values[start:start + VALS_PER_LINE]
        end = start + len(chunk)
        line = ",".join(VAL_FMT % v for v in chunk)
        if end < n:
            line += ","
        out.append(line)


def emit_weight(name, data, out):
    """2-D table: inline const double name[out_features][in_features]."""
    n_out, n_in = data.shape
    out.append("inline const double %s[%d][%d] = {" % (name, n_out, n_in))
    for j in range(n_out):
        rows = []
        format_values(data[j, :], rows)
        rows[0] = "{" + rows[0]
        rows[-1] = rows[-1] + "},"
        out.extend(rows)
    out.append("};")


def emit_bias(name, data, out):
    """1-D table: inline const double name[features]."""
    out.append("inline const double %s[%d] = {" % (name, data.shape[0]))
    format_values(data, out)
    out.append("};")


# ---------------------------------------------------------------------------
#  Model introspection
# ---------------------------------------------------------------------------
def extract_layers(model):
    """Return [(weight, bias), ...] in forward order as float64 arrays.

    Parameters are assumed to alternate weight, bias, weight, bias, ... which
    is what a plain stack of nn.Linear layers produces.  Unlike the Fortran
    generator, the weights are kept in PyTorch's (out, in) orientation: the C++
    `dense` helper indexes them as w[j][i].
    """
    params = [p.detach().double().numpy() for p in model.parameters()]
    if len(params) == 0 or len(params) % 2 != 0:
        raise ValueError(
            "expected an even, non-zero number of parameters (weight/bias "
            "pairs), got %d" % len(params)
        )

    layers = []
    for i in range(0, len(params), 2):
        weight, bias = params[i], params[i + 1]
        if weight.ndim != 2 or bias.ndim != 1:
            raise ValueError("parameter %d is not a (weight, bias) pair" % i)
        if weight.shape[0] != bias.shape[0]:
            raise ValueError(
                "layer %d: weight rows (%d) do not match bias length (%d)"
                % (i // 2 + 1, weight.shape[0], bias.shape[0])
            )
        layers.append((weight, bias))

    for k in range(1, len(layers)):
        if layers[k][0].shape[1] != layers[k - 1][0].shape[0]:
            raise ValueError(
                "layer %d input (%d) does not match layer %d output (%d)"
                % (k + 1, layers[k][0].shape[1], k, layers[k - 1][0].shape[0])
            )
    return layers


# ---------------------------------------------------------------------------
#  Forward pass (architecture dependent)
# ---------------------------------------------------------------------------
def emit_get_class(layers, out):
    n_out = layers[-1][0].shape[0]
    hidden = [w.shape[0] for w, _ in layers[:-1]]

    out.append("inline int get_class(Stencil& s) {")
    out.append("  if (s.f(CID, CID, CID) < EPSILON_CONNECT) return 0;")
    out.append("")
    out.append("  double flat[NIN];")
    out.append("  detail::preprocess_and_flatten(s, flat);")
    out.append("")
    decls = ["h%d[%d]" % (i + 1, size) for i, size in enumerate(hidden)]
    decls.append("logits[%d]" % n_out)
    out.append("  double %s;" % ", ".join(decls))

    src = "flat"
    for i in range(len(layers)):
        last = i == len(layers) - 1
        dst = "logits" if last else "h%d" % (i + 1)
        out.append(
            "  detail::dense(%s, detail::lay%d_weight, detail::lay%d_bias, "
            "%s, %s);" % (src, i + 1, i + 1, dst, "false" if last else "true")
        )
        src = dst
    out.append("")
    out.append("  // maxloc: first occurrence of the maximum, 1-based")
    out.append("  int best = 0;")
    out.append("  for (int i = 1; i < %d; ++i)" % n_out)
    out.append("    if (logits[i] > logits[best]) best = i;")
    out.append("  return best + 1;")
    out.append("}")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
def generate(model, ncell=125):
    layers = extract_layers(model)

    n_in = layers[0][0].shape[1]
    expected = ncell * 4
    if n_in != expected:
        raise ValueError(
            "model input size %d does not match the %d-cell stencil layout "
            "(expected %d)" % (n_in, ncell, expected)
        )

    arch = " -> ".join([str(n_in)] + [str(w.shape[0]) for w, _ in layers])

    out = []
    out.append(HEADER_TOP.replace("{arch}", arch).rstrip("\n"))
    out.append("")

    for i, (weight, bias) in enumerate(layers):
        emit_weight("lay%d_weight" % (i + 1), weight, out)
        emit_bias("lay%d_bias" % (i + 1), bias, out)

    out.append(BODY_TOP.rstrip("\n"))
    emit_get_class(layers, out)
    out.append(BODY_BOTTOM.rstrip("\n"))
    out.append("")
    out.append(FOOTER)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ml_classifier.h from a TorchScript model."
    )
    parser.add_argument("model", nargs="?", default="./ml_model_aug6.pt",
                        help="TorchScript model file (default: %(default)s)")
    parser.add_argument("-o", "--output", default="ml_classifier.h",
                        help="output header (default: %(default)s)")
    args = parser.parse_args()

    model = torch.jit.load(args.model)
    model.eval()
    text = generate(model)

    # The reference header ends without a trailing newline; match it exactly.
    with open(args.output, "w") as fh:
        fh.write(text)

    print("wrote %s (%d lines)" % (args.output, text.count("\n") + 1))


if __name__ == "__main__":
    main()