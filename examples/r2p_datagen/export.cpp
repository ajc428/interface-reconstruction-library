// Writes a trained R2P-Net (trainer.cpp's model.pt) into an r2pnet.h, without
// Python: loads the model with the trainer's own IRL::model class and
// torch::load, then replaces every layN_weight / layN_bias array of a template
// header (the current r2pnet.h, which also supplies get_normals and the
// reflection code) with the trained values.
//
//   r2p_export MODEL.pt TEMPLATE_r2pnet.h   OUT.h   [in out depth width]
//   r2p_export MODEL.pt TEMPLATE_r2pnet.f90 OUT.f90 [in out depth width]
//   r2p_export MODEL.pt --forward MOMENTS N         [in out depth width]
//
// The .f90 form rewrites only the DATA statements of a generate_r2pnet.py
// module (transposed weights, d-exponent literals that round-trip exactly),
// keeping its declarations, get_normals and reflection code.
//       prints the libtorch outputs for the first N rows of MOMENTS, to check
//       an exported header (or a result_pr.txt) against the model itself
//
// in/out/depth/width default to 192 6 3 256 (trainer type 1). Every array's
// shape is checked against the template, so a model of a different size is
// refused rather than silently mis-exported.

#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "irl/machine_learning_reconstruction/neural_network.h"

// Shortest-exact double literal with a d exponent (DEFAULT REAL literals in a
// DATA statement would silently lose ~7 digits).
static std::string fortranDouble(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  std::string s = buf;
  const auto e = s.find_first_of("eE");
  if (e != std::string::npos) s[e] = 'd';
  else s += "d0";
  return s;
}

// DATA statement in generate_r2pnet.py's layout: wrapped on commas, '&' continuations.
static void dataStatement(std::string& out, const std::string& name, const std::string& index, const double* v,
                          long n) {
  std::string values;
  for (long k = 0; k < n; ++k) values += (k ? ", " : "") + fortranDouble(v[k]);
  out += "   DATA (" + name + "(" + index + "), idx=1, " + std::to_string(n) + ") /&\n";
  const std::size_t max_len = 1900;
  while (values.size() > max_len) {
    std::size_t cut = values.rfind(',', max_len);
    if (cut == std::string::npos) cut = max_len;
    out += "   " + values.substr(0, cut + 1) + "&\n";
    values = values.substr(values.find_first_not_of(' ', cut + 1));
  }
  out += "   " + values + "/\n";
}

static int writeFortran(const std::vector<torch::Tensor>& params, const char* tpath, const char* opath) {
  std::ifstream tin(tpath);
  if (!tin) { std::fprintf(stderr, "cannot open template %s\n", tpath); return 2; }
  std::stringstream ss;
  ss << tin.rdbuf();
  const std::string text = ss.str();
  // Declared shapes must match: weights are (in, out) in Fortran.
  const std::regex decl(R"(real\(WP\), dimension\((\d+)(?:,(\d+))?\), save :: lay(\d+)_(weight|bias))");
  int checked = 0;
  for (std::sregex_iterator it(text.begin(), text.end(), decl), end; it != end; ++it) {
    const std::smatch& m = *it;
    const std::size_t idx = 2 * (std::stoi(m[3]) - 1) + (m[4] == "weight" ? 0 : 1);
    if (idx >= params.size()) { std::fprintf(stderr, "template declares lay%s_%s, model has no such layer\n", m[3].str().c_str(), m[4].str().c_str()); return 1; }
    const auto& t = params[idx];
    const bool ok = m[4] == "weight" ? (t.dim() == 2 && t.size(1) == std::stol(m[1]) && t.size(0) == std::stol(m[2]))
                                     : (t.dim() == 1 && t.size(0) == std::stol(m[1]));
    if (!ok) { std::fprintf(stderr, "shape mismatch for lay%s_%s\n", m[3].str().c_str(), m[4].str().c_str()); return 1; }
    ++checked;
  }
  if (checked != int(params.size())) { std::fprintf(stderr, "template declares %d arrays, model has %zu\n", checked, params.size()); return 1; }
  const std::size_t first = text.find("   DATA (");
  const std::size_t contains = text.find("\n   contains", first);
  if (first == std::string::npos || contains == std::string::npos) { std::fprintf(stderr, "no DATA block in template\n"); return 1; }
  std::string data;
  for (std::size_t p = 0; p < params.size(); ++p) {
    const std::string name = "lay" + std::to_string(p / 2 + 1) + (p % 2 == 0 ? "_weight" : "_bias");
    const auto& t = params[p];
    const double* v = t.data_ptr<double>();
    if (p % 2 == 0) {
      // Column j of the Fortran (in,out) array = output neuron j's weights = row j of the model's matrix.
      const long rows = t.size(0), cols = t.size(1);
      for (long r = 0; r < rows; ++r) dataStatement(data, name, "idx, " + std::to_string(r + 1), v + r * cols, cols);
    } else {
      dataStatement(data, name, "idx", v, t.size(0));
    }
  }
  std::ofstream(opath) << text.substr(0, first) << data << text.substr(contains);
  std::printf("wrote %s: %zu arrays\n", opath, params.size());
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: r2p_export MODEL.pt TEMPLATE_r2pnet.h OUT.h [in out depth width]\n");
    return 2;
  }
  // Both forms take three arguments after MODEL.pt, then the optional sizes.
  const bool forward_mode = std::string(argv[2]) == "--forward";
  if (forward_mode && argc < 5) {
    std::fprintf(stderr, "usage: r2p_export MODEL.pt --forward MOMENTS N [in out depth width]\n");
    return 2;
  }
  auto size = [&](int k, int fallback) { return argc > 5 + k ? std::stoi(argv[5 + k]) : fallback; };
  const int in = size(0, 192), out = size(1, 6), depth = size(2, 3), width = size(3, 256);

  auto nn = std::make_shared<IRL::model>(in, out, depth, width, 1);
  torch::load(nn, argv[1]);
  nn->eval();

  if (forward_mode) {
    torch::NoGradGuard no_grad;
    const auto dtype = nn->parameters()[0].scalar_type();
    std::ifstream fin(argv[3]);
    const long N = std::stol(argv[4]);
    std::string line, tok;
    for (long n = 0; n < N && std::getline(fin, line); ++n) {
      std::vector<double> x;
      std::stringstream ls(line);
      while (std::getline(ls, tok, ',')) x.push_back(std::stod(tok));
      const auto y = nn->forward(torch::tensor(x, torch::kFloat64).to(dtype).unsqueeze(0)).squeeze(0).to(torch::kFloat64);
      for (int k = 0; k < out; ++k) std::printf(k + 1 < out ? "%.10g " : "%.10g\n", y[k].item<double>());
    }
    return 0;
  }

  // Parameters in registration order: l1.weight, l1.bias, l2..., output layer.
  std::vector<torch::Tensor> params;
  for (const auto& p : nn->named_parameters()) params.push_back(p.value().detach().to(torch::kFloat64).contiguous());

  const std::string out_path = argv[3];
  if (out_path.size() > 4 && out_path.compare(out_path.size() - 4, 4, ".f90") == 0) return writeFortran(params, argv[2], argv[3]);

  std::ifstream tin(argv[2]);
  if (!tin) { std::fprintf(stderr, "cannot open template %s\n", argv[2]); return 2; }
  std::stringstream ss;
  ss << tin.rdbuf();
  std::string text = ss.str();

  const std::regex decl(R"(const double (lay(\d+)_(weight|bias))\[(\d+)\](?:\[(\d+)\])? = \{)");
  std::string result;
  std::size_t pos = 0;
  int replaced = 0;
  for (std::sregex_iterator it(text.begin(), text.end(), decl), end; it != end; ++it) {
    const std::smatch& m = *it;
    const int layer = std::stoi(m[2]);
    const bool weight = m[3] == "weight";
    const int rows = std::stoi(m[4]), cols = m[5].matched ? std::stoi(m[5]) : 0;
    const std::size_t idx = 2 * (layer - 1) + (weight ? 0 : 1);
    if (idx >= params.size()) { std::fprintf(stderr, "template has %s but the model has no such layer\n", m[1].str().c_str()); return 1; }
    const torch::Tensor& t = params[idx];
    const bool ok = weight ? (t.dim() == 2 && t.size(0) == rows && t.size(1) == cols) : (t.dim() == 1 && t.size(0) == rows);
    if (!ok) {
      std::fprintf(stderr, "shape mismatch for %s: template [%d]%s, model %s\n", m[1].str().c_str(), rows,
                   weight ? ("[" + std::to_string(cols) + "]").c_str() : "", c10::str(t.sizes()).c_str());
      return 1;
    }
    const std::size_t start = m.position(0);
    const std::size_t close = text.find("};", start);
    result += text.substr(pos, start - pos);
    std::string block = m.str(0) + "\n";
    const double* v = t.data_ptr<double>();
    char buf[40];
    if (weight) {
      for (int r = 0; r < rows; ++r) {
        block += "    {";
        for (int c = 0; c < cols; ++c) {
          std::snprintf(buf, sizeof(buf), c ? ", %.17g" : "%.17g", v[r * cols + c]);
          block += buf;
        }
        block += r + 1 < rows ? "},\n" : "}\n";
      }
    } else {
      block += "    ";
      for (int r = 0; r < rows; ++r) {
        std::snprintf(buf, sizeof(buf), r ? ", %.17g" : "%.17g", v[r]);
        block += buf;
      }
      block += "\n";
    }
    result += block;
    pos = close;
    ++replaced;
  }
  result += text.substr(pos);
  if (replaced != int(params.size())) {
    std::fprintf(stderr, "template has %d arrays, model has %zu parameter tensors\n", replaced, params.size());
    return 1;
  }
  std::ofstream(argv[3]) << result;
  std::printf("wrote %s: %d arrays from %s\n", argv[3], replaced, argv[1]);
  return 0;
}
