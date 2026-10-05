// Scores R2P-Net predictions against generator labels, broken down by the
// generator's metadata.
//
//   r2p_eval --meta meta_test.txt --labels normals_test.txt
//            [--pred NAME=FILE ...]          predictions, one line per row in
//                                            file order (e.g. trainer.cpp's
//                                            result_pr.txt; spaces or commas)
//            [--deployed NAME --moments moments_test.txt]
//                                            the network compiled into r2pnet.h
//            [--by family,radius,thickness,tip,splay,vf,area | all]
//            [--worst K]                     K worst samples per family (for
//                                            r2p_datagen CONFIG --viz)
//            [--rows FILE]                   per-row errors of every source
//
// Labels in either generator format: 6 values (absent face = zeros) or 8
// (n1 n2 p1 p2, p = 1 present / 0 absent / -1 ignore). Predictions with 8 or
// more values use values 7-8 as presence logits; otherwise a slot counts as a
// plane if |n| >= 0.85, as R2P3D_Net decides today.
//
// Per-face error = angle between the normalized prediction and the label, for
// every face the label has (present or ignore). "swap" = share of two-face
// samples whose error would drop by more than 5 degrees if the two slots were
// exchanged: a slot-ordering failure rather than a direction error.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "examples/new_advector/r2pnet.h"

namespace {

struct Row {
  std::map<std::string, std::string> meta;
  double label[8] = {0};
  int nlabel = 0;
  double num(const std::string& k) const {
    auto it = meta.find(k);
    return it == meta.end() || it->second.empty() ? std::nan("") : std::stod(it->second);
  }
};

struct Pred {
  std::string name;
  std::vector<std::vector<double>> v;
};

std::vector<double> parseLine(const std::string& line) {
  std::vector<double> out;
  std::string tok;
  for (char c : line + " ") {
    if (c == ',' || c == ' ' || c == '\t' || c == '\r') {
      if (!tok.empty()) { out.push_back(std::stod(tok)); tok.clear(); }
    } else {
      tok += c;
    }
  }
  return out;
}

double mag(const double* v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
double angle(const double* a, const double* b) {
  const double ma = mag(a), mb = mag(b);
  if (ma < 1e-12 || mb < 1e-12) return 180.0;
  const double c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (ma * mb);
  return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

// Label presence per slot: 1 present, 0 absent, -1 ignore.
int labelPresence(const Row& r, int k) {
  if (r.nlabel >= 8) return int(r.label[6 + k]);
  return mag(r.label + 3 * k) > 0.5 ? 1 : 0;
}
bool predPresent(const std::vector<double>& p, int k) {
  if (p.size() >= 8) return p[6 + k] > 0.0;   // logit > 0  <=>  sigmoid > 0.5
  return mag(p.data() + 3 * k) >= 0.85;
}

struct Score {
  std::vector<double> face;   // per-face errors
  long n = 0, count_ok = 0, count_n = 0, swap = 0, two = 0;
  long l1p2 = 0, l2p1 = 0;    // label 1 plane -> predicted 2, and the reverse
};

// Errors of one row for one source; returns the row's worst face error.
double scoreRow(const Row& r, const std::vector<double>& p, Score* s) {
  ++s->n;
  double worst = 0.0, sum_straight = 0.0, sum_swapped = 0.0;
  int nl = 0;
  for (int k = 0; k < 2; ++k) {
    if (mag(r.label + 3 * k) < 0.5) continue;
    const double e = angle(p.data() + 3 * k, r.label + 3 * k);
    s->face.push_back(e);
    worst = std::max(worst, e);
    sum_straight += e;
    sum_swapped += angle(p.data() + 3 * (1 - k), r.label + 3 * k);
    ++nl;
  }
  if (nl == 2) {
    ++s->two;
    if (sum_swapped + 10.0 < sum_straight) ++s->swap;   // > 5 deg per face on average
  }
  const int l0 = labelPresence(r, 0), l1 = labelPresence(r, 1);
  if (l0 >= 0 && l1 >= 0) {   // count metric skips ignore-band labels
    const int want = (l0 == 1) + (l1 == 1);
    const int got = std::max(1, int(predPresent(p, 0)) + int(predPresent(p, 1)));
    ++s->count_n;
    s->count_ok += want == got;
    s->l1p2 += want == 1 && got == 2;
    s->l2p1 += want == 2 && got == 1;
  }
  return worst;
}

std::string summary(Score s) {
  if (s.face.empty()) return "      -";
  std::sort(s.face.begin(), s.face.end());
  const std::size_t m = s.face.size();
  auto at = [&](double q) { return s.face[std::min(m - 1, std::size_t(q * (m - 1) + 0.5))]; };
  long gt5 = 0, gt10 = 0;
  for (double e : s.face) { gt5 += e > 5.0; gt10 += e > 10.0; }
  char b[256];
  std::snprintf(b, sizeof(b), "%6.2f %6.2f %6.2f %5.1f%% %5.1f%% | %5.1f%% %5.2f%% %5.2f%% %5.2f%%", at(0.5), at(0.9),
                at(0.99), 100.0 * gt5 / m, 100.0 * gt10 / m,
                s.count_n ? 100.0 * s.count_ok / s.count_n : 0.0, s.count_n ? 100.0 * s.l1p2 / s.count_n : 0.0,
                s.count_n ? 100.0 * s.l2p1 / s.count_n : 0.0, s.two ? 100.0 * s.swap / s.two : 0.0);
  return b;
}

const char* kSummaryHeader = "   med    p90    p99    >5deg  >10deg | count  1->2   2->1   swap";

struct Breakdown {
  std::string name, title;
  std::function<std::string(const Row&)> key;   // "" = skip the row
};

std::string bin(double x, const std::vector<double>& edges, const char* fmt = "%g") {
  if (std::isnan(x)) return "";
  for (std::size_t i = 0; i + 1 < edges.size(); ++i)
    if (x >= edges[i] && x < edges[i + 1]) {
      char b[64], lo[24], hi[24];
      std::snprintf(lo, sizeof(lo), fmt, edges[i]);
      std::snprintf(hi, sizeof(hi), fmt, edges[i + 1]);
      std::snprintf(b, sizeof(b), "%02zu [%s, %s)", i, lo, hi);
      return b;
    }
  return "";
}

std::vector<Breakdown> breakdowns() {
  const double inf = INFINITY;
  return {
      {"family", "by family", [](const Row& r) { return r.meta.at("family"); }},
      {"radius", "by smallest radius 1/(2 max(|a|,|b|)) (cells; sheets)",
       [inf](const Row& r) {
         const double a = std::abs(r.num("a")), b = std::abs(r.num("b"));
         if (std::isnan(a) || r.meta.at("family") == "edge") return std::string();
         const double m = std::max(a, b);
         return bin(m > 0 ? 0.5 / m : inf, {0, 5, 10, 20, 50, inf});
       }},
      {"thickness", "by film thickness at the anchor (cells)",
       [inf](const Row& r) { return bin(r.num("thickness"), {0, 0.05, 0.1, 0.3, 1, 1.7320508, inf}, "%.3g"); }},
      {"tip", "by tip location (edges; 0 centre cell, 1 stencil, 2 outside)",
       [](const Row& r) {
         const double z = r.num("tip_zone");
         return (std::isnan(z) || z < 0 || r.meta.at("family") == "wedge") ? std::string()
                                                                            : "zone " + std::to_string(int(z));
       }},
      {"splay", "by angle between the two labelled faces (deg; two-face rows)",
       [](const Row& r) { return bin(r.num("label_splay"), {0, 1, 5, 15, 45, 180}); }},
      {"vf", "by phase-0 centre-cell VF (the network's phase)",
       [](const Row& r) {
         double v = r.num("vf_centre");
         if (r.num("phase0_gas") == 1) v = 1.0 - v;
         return bin(v, {0, 0.016, 0.1, 0.5, 0.9, 1.0000001}, "%.3g");
       }},
      {"area", "by smaller face area (cell-face units; two-face rows)",
       [inf](const Row& r) {
         if (r.meta.at("nfaces") != "2") return std::string();
         return bin(std::min(r.num("area1"), r.num("area2")), {0, 0.01, 0.05, 0.2, 1, inf});
       }},
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::string meta_path, labels_path, moments_path, deployed_name, rows_path, by = "family,radius,thickness,tip";
  std::vector<std::pair<std::string, std::string>> pred_files;
  int worst = 0;
  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto next = [&]() { if (a + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(2); } return std::string(argv[++a]); };
    if (k == "--meta") meta_path = next();
    else if (k == "--labels") labels_path = next();
    else if (k == "--moments") moments_path = next();
    else if (k == "--deployed") deployed_name = next();
    else if (k == "--pred") {
      const std::string v = next();
      const auto eq = v.find('=');
      if (eq == std::string::npos) { std::fprintf(stderr, "--pred NAME=FILE\n"); return 2; }
      pred_files.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    }
    else if (k == "--by") by = next();
    else if (k == "--worst") worst = std::stoi(next());
    else if (k == "--rows") rows_path = next();
    else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
  }
  if (meta_path.empty() || labels_path.empty() || (pred_files.empty() && deployed_name.empty())) {
    std::fprintf(stderr, "usage: r2p_eval --meta META --labels LABELS [--pred NAME=FILE ...] "
                         "[--deployed NAME --moments MOMENTS] [--by LIST|all] [--worst K] [--rows FILE]\n");
    return 2;
  }
  if (by == "all") by = "family,radius,thickness,tip,splay,vf,area";

  // Metadata and labels.
  std::vector<Row> rows;
  {
    std::ifstream fm(meta_path), fl(labels_path);
    if (!fm || !fl) { std::fprintf(stderr, "cannot open meta or labels\n"); return 2; }
    std::string header, line;
    std::getline(fm, header);
    std::vector<std::string> cols;
    std::stringstream hs(header);
    for (std::string c; std::getline(hs, c, ',');) cols.push_back(c);
    while (std::getline(fm, line)) {
      Row r;
      std::stringstream ss(line);
      std::string v;
      for (std::size_t c = 0; c < cols.size() && std::getline(ss, v, ','); ++c) r.meta[cols[c]] = v;
      std::string ll;
      if (!std::getline(fl, ll)) { std::fprintf(stderr, "labels file shorter than meta\n"); return 2; }
      const auto lab = parseLine(ll);
      r.nlabel = int(std::min<std::size_t>(8, lab.size()));
      for (int k = 0; k < r.nlabel; ++k) r.label[k] = lab[k];
      rows.push_back(std::move(r));
    }
  }
  const std::size_t N = rows.size();

  // Prediction sources.
  std::vector<Pred> preds;
  for (const auto& [name, path] : pred_files) {
    Pred p{name, {}};
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 2; }
    for (std::string line; std::getline(in, line) && p.v.size() < N;)
      if (!line.empty()) p.v.push_back(parseLine(line));
    if (p.v.size() < N) {
      std::fprintf(stderr, "note: %s has %zu rows, meta %zu: scoring the first %zu\n", name.c_str(), p.v.size(), N,
                   p.v.size());
    }
    preds.push_back(std::move(p));
  }
  if (!deployed_name.empty()) {
    if (moments_path.empty()) { std::fprintf(stderr, "--deployed needs --moments\n"); return 2; }
    Pred p{deployed_name, {}};
    std::ifstream in(moments_path);
    for (std::string line; std::getline(in, line) && p.v.size() < N;) {
      const auto x = parseLine(line);
      double out[6];
      r2pnet::get_normals(x.data(), out);
      p.v.emplace_back(out, out + 6);
    }
    preds.push_back(std::move(p));
  }
  std::size_t n_eval = N;
  for (const auto& p : preds) n_eval = std::min(n_eval, p.v.size());
  std::printf("%zu rows scored (meta %s)\n", n_eval, meta_path.c_str());

  // Per-row worst-face errors, per source.
  std::vector<std::vector<double>> row_err(preds.size(), std::vector<double>(n_eval, 0.0));
  for (std::size_t s = 0; s < preds.size(); ++s)
    for (std::size_t i = 0; i < n_eval; ++i) {
      Score dummy;
      row_err[s][i] = scoreRow(rows[i], preds[s].v[i], &dummy);
    }

  auto table = [&](const Breakdown& b) {
    std::map<std::string, std::vector<Score>> groups;
    for (std::size_t i = 0; i < n_eval; ++i) {
      const std::string key = b.key(rows[i]);
      if (key.empty()) continue;
      auto& g = groups[key];
      if (g.empty()) g.resize(preds.size());
      for (std::size_t s = 0; s < preds.size(); ++s) scoreRow(rows[i], preds[s].v[i], &g[s]);
    }
    if (groups.empty()) return;
    std::printf("\n%s  (per-face angle error in deg; count = plane count agrees; 1->2, 2->1 = wrong counts; "
                "swap = slot order)\n", b.title.c_str());
    std::printf("  %-22s %-10s %7s  %s\n", "", "source", "rows", kSummaryHeader);
    for (const auto& [key, scores] : groups) {
      std::string label = key;
      if (label.size() > 3 && label[2] == ' ' && std::isdigit(label[0])) label = label.substr(3);   // drop sort prefix
      for (std::size_t s = 0; s < preds.size(); ++s)
        std::printf("  %-22s %-10s %7ld  %s\n", s == 0 ? label.c_str() : "", preds[s].name.c_str(), scores[s].n,
                    summary(scores[s]).c_str());
    }
  };

  {
    Breakdown overall{"all", "overall", [](const Row&) { return std::string("all"); }};
    table(overall);
  }
  const auto all = breakdowns();
  std::stringstream bs(by);
  for (std::string name; std::getline(bs, name, ',');) {
    auto it = std::find_if(all.begin(), all.end(), [&](const Breakdown& b) { return b.name == name; });
    if (it == all.end()) { std::fprintf(stderr, "unknown breakdown %s\n", name.c_str()); return 2; }
    table(*it);
  }

  if (worst > 0) {
    for (std::size_t s = 0; s < preds.size(); ++s) {
      std::printf("\nworst %d per family for %s (sample = generator index; view with r2p_datagen CONFIG --viz SPLIT "
                  "I,I,...)\n", worst, preds[s].name.c_str());
      std::map<std::string, std::vector<std::size_t>> fam;
      for (std::size_t i = 0; i < n_eval; ++i) fam[rows[i].meta.at("family")].push_back(i);
      for (auto& [f, idx] : fam) {
        std::partial_sort(idx.begin(), idx.begin() + std::min<std::size_t>(worst, idx.size()), idx.end(),
                          [&](std::size_t a, std::size_t b) { return row_err[s][a] > row_err[s][b]; });
        std::string list;
        std::printf("  %s:\n", f.c_str());
        for (std::size_t k = 0; k < std::min<std::size_t>(worst, idx.size()); ++k) {
          const Row& r = rows[idx[k]];
          const std::string sample = r.meta.count("sample") ? r.meta.at("sample") : std::to_string(idx[k]);
          std::printf("    sample %-8s err %6.2f  nfaces %s  thickness %-8s splay %-8s tip_zone %s\n", sample.c_str(),
                      row_err[s][idx[k]], r.meta.at("nfaces").c_str(), r.meta.at("thickness").c_str(),
                      r.meta.at("label_splay").c_str(), r.meta.count("tip_zone") ? r.meta.at("tip_zone").c_str() : "-");
          list += (list.empty() ? "" : ",") + sample;
        }
        std::printf("    --viz list: %s\n", list.c_str());
      }
    }
  }

  if (!rows_path.empty()) {
    std::ofstream out(rows_path);
    out << "row,sample,family";
    for (const auto& p : preds) out << ',' << p.name;
    out << '\n';
    for (std::size_t i = 0; i < n_eval; ++i) {
      out << i << ',' << (rows[i].meta.count("sample") ? rows[i].meta.at("sample") : std::to_string(i)) << ','
          << rows[i].meta.at("family");
      for (std::size_t s = 0; s < preds.size(); ++s) out << ',' << row_err[s][i];
      out << '\n';
    }
  }
  return 0;
}
