// R2P-Net training-data generator.
//
//   mpirun -np P r2p_datagen CONFIG           generate the splits in CONFIG
//   r2p_datagen CONFIG --selftest             check moments against Monte
//                                             Carlo and the label transform
//                                             against R2P3D_Net's inverse
//   r2p_datagen CONFIG --viz SPLIT I[,I...]   ParaView files for samples I of
//                                             SPLIT (train|val|test), exactly as
//                                             generated (e.g. worst cases from
//                                             r2p_eval)
//
// Output per split (train -> "", val -> "_val", test -> "_test"), in
// output_dir, in the formats trainer.cpp reads:
//   moments<s>.txt   192 network inputs per line
//   normals<s>.txt   labels per line: 6 (legacy: n1 n2, absent face = zeros)
//                    or 8 (presence: n1 n2 p1 p2), see features.h
//   meta<s>.txt      CSV with header: family, parameters, truth, face areas
//   moments5<s>.txt  raw 5^3 liquid moments (7 per cell), if stencil = 5
//   viz/             ParaView files for the first viz.per_family samples of
//                    each family in the first split, if viz.per_family > 0
//
// Reproducible and independent of the rank count: sample i of a split uses
// its own random stream seeded from (seed, split, i), its family is fixed by
// i alone (low-discrepancy assignment, so the family shares are exact to
// O(log N / N)), ranks take contiguous blocks of i, and the shards are merged
// in rank order.

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "examples/r2p_datagen/config.h"
#include "examples/r2p_datagen/features.h"
#include "examples/r2p_datagen/unpinch.h"
#include "examples/r2p_datagen/scene.h"
#include "examples/r2p_datagen/stencil.h"
#include "examples/r2p_datagen/viz.h"

using namespace r2pgen;

namespace {

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

struct Setup {
  std::vector<Family> families;
  std::vector<double> cumulative;   // normalized cumulative weights
  Placement placement;
  FeatureSettings features;
  int stencil = 3;
  double noise_sigma = 0.0125;
  bool noise_consistent = false;
  double vf_floor = 1.0e-9;
  double label_min_gap_fraction = 0.0;   // > 0: labels may not pinch a continuing film (unpinch.h)
  int max_attempts = 1000;
  std::uint64_t seed = 1;
  std::string output_dir = ".";
  struct Split { std::string name, suffix; int id; long count; };
  std::vector<Split> splits;

  // Random stream of sample i of a split. The split id is fixed (train 0,
  // val 1, test 2), so a split's samples do not depend on which others are on.
  std::uint64_t stream(int split_id, long i) const {
    return seed * 0x100000001B3ULL + std::uint64_t(split_id) * 0x9E3779B97F4AULL + std::uint64_t(i);
  }
  int viz_per_family = 0;
  std::string viz_dir = "viz";
  double viz_length = 0.1;

  int familyOf(long i) const {
    const double golden = 0.6180339887498949;
    const double u = std::fmod((i + 0.5) * golden, 1.0);
    for (std::size_t f = 0; f < cumulative.size(); ++f)
      if (u < cumulative[f]) return int(f);
    return int(cumulative.size()) - 1;
  }

  static Setup fromConfig(const Config& cfg) {
    Setup s;
    const Section& g = cfg.global;
    s.seed = std::uint64_t(g.num("seed", 1));
    s.output_dir = g.str("output_dir", ".");
    s.stencil = int(g.num("stencil", 3));
    if (s.stencil != 3 && s.stencil != 5) throw std::runtime_error("stencil must be 3 or 5");
    s.noise_sigma = g.num("noise.sigma", 0.0125);
    const std::string nm = g.str("noise.mode", "independent");
    if (nm != "independent" && nm != "consistent") throw std::runtime_error("noise.mode: independent | consistent");
    s.noise_consistent = nm == "consistent";
    s.vf_floor = g.num("vf_floor", 1.0e-9);
    s.max_attempts = int(g.num("max_attempts", 1000));
    s.placement.band_fraction = g.num("placement.band_fraction", 0.5);
    s.placement.aligned_fraction = g.num("rotation.aligned_fraction", 0.0);
    s.placement.jitter_deg = g.num("rotation.jitter", 10.0);
    const std::string p0 = g.str("phase0", "pca");
    s.features.phase0 = p0 == "pca" ? Phase0Rule::kPCA : p0 == "film" ? Phase0Rule::kFilm
                      : p0 == "vfsum" ? Phase0Rule::kVFSum : throw std::runtime_error("phase0: pca | film | vfsum");
    const std::string lo = g.str("label.order", "xsign");
    s.features.order = lo == "xsign" ? LabelOrder::kXSign : lo == "pca" ? LabelOrder::kPCADot
                     : throw std::runtime_error("label.order: xsign | pca");
    const std::string lf = g.str("label.format", "legacy");
    s.features.format = lf == "legacy" ? LabelFormat::kLegacy : lf == "presence" ? LabelFormat::kPresence
                      : throw std::runtime_error("label.format: legacy | presence");
    s.features.min_area = g.num("label.min_area", 1.0e-4);
    s.label_min_gap_fraction = g.num("label.min_gap_fraction", 0.0);
    s.features.absent_area = g.num("label.absent_area", 0.01);
    s.features.present_area = g.num("label.present_area", 0.05);
    s.viz_per_family = int(g.num("viz.per_family", 0));
    s.viz_dir = g.str("viz.dir", "viz");
    s.viz_length = g.num("viz.resolution", 0.1);
    int id = 0;
    for (const auto& [suffix, key] : {std::pair<std::string, std::string>{"", "train"}, {"_val", "val"}, {"_test", "test"}}) {
      const long n = long(g.num(key, 0));
      if (n > 0) s.splits.push_back({key, suffix, id, n});
      ++id;
    }
    double total = 0.0;
    for (const auto& sec : cfg.families) {
      s.families.push_back(Family::fromSection(sec));
      total += s.families.back().weight;
    }
    if (s.families.empty() || !(total > 0.0)) throw std::runtime_error("config defines no families with weight > 0");
    double acc = 0.0;
    for (const auto& f : s.families) s.cumulative.push_back((acc += f.weight) / total);
    cfg.checkAllUsed();
    return s;
  }
};

struct Sample {
  Scene scene;
  std::vector<double> moments;   // raw liquid moments, stencil^3, after noise
  Features features;
  int attempts = 0;
  int tip_zone = -1;
  double unpinch = 0.0;   // label rotation toward the mean film normal (0 = unchanged), unpinch.h
};

template <class Engine>
void addNoise(std::vector<double>& m, int N, const Setup& s, Engine& eng) {
  if (s.noise_sigma <= 0.0) return;
  std::normal_distribution<double> noise(0.0, s.noise_sigma);
  auto clip = [](double x) { return std::min(0.5, std::max(-0.5, x)); };
  for (int c = 0; c < N * N * N; ++c) {
    double* o = &m[7 * c];
    if (o[0] <= IRL::global_constants::VF_LOW || o[0] >= IRL::global_constants::VF_HIGH) continue;
    for (int d = 0; d < 3; ++d) o[1 + d] = clip(o[1 + d] + noise(eng));
    for (int d = 0; d < 3; ++d)
      o[4 + d] = s.noise_consistent ? clip(-o[0] * o[1 + d] / (1.0 - o[0])) : clip(o[4 + d] + noise(eng));
  }
}

// The 3^3 block at the centre of an N^3 stencil.
std::vector<double> centre3(const std::vector<double>& m, int N) {
  if (N == 3) return m;
  std::vector<double> out(7 * 27);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int v = 0; v < 7; ++v)
          out[7 * (i * 9 + j * 3 + k) + v] = m[7 * ((i + 1) * 25 + (j + 1) * 5 + (k + 1)) + v];
  return out;
}

// One sample; false only if max_attempts draws all failed.
bool generate(const Setup& s, int family, std::uint64_t stream, Sample* out) {
  std::mt19937_64 eng(splitmix64(stream));
  const Family& fam = s.families[family];
  const int N = s.stencil;
  for (int attempt = 1; attempt <= s.max_attempts; ++attempt) {
    Scene canon;
    if (!fam.build(eng, &canon)) continue;
    Scene sc = s.placement.place(canon, eng);
    sc.truth.family = family;
    if (sc.kind == Kind::kNested && !nestedInBox(sc.upper, sc.lower, 0.5 * N)) continue;
    std::vector<double> m = stencilMoments(sc, N, s.vf_floor);
    const int centre = (N * N * N) / 2;
    if (m[7 * centre] <= IRL::global_constants::VF_LOW || m[7 * centre] >= IRL::global_constants::VF_HIGH) continue;
    Faces faces = centreFaces(sc);
    addNoise(m, N, s, eng);
    // After the noise: the planes are placed from the centroids the network
    // (and the deployed Newton solve) will see.
    const double unpinch = s.label_min_gap_fraction > 0.0
                               ? unpinchLabels(sc, &m[7 * centre], s.label_min_gap_fraction, &faces)
                               : 0.0;
    const std::vector<double> m3 = centre3(m, N);
    const int film_is_gas = fam.kind == FamilyKind::kBulk ? -1 : 0;   // films are liquid
    out->features = buildFeatures(m3.data(), faces, film_is_gas, s.features);
    if (out->features.nfaces == 0) continue;
    out->scene = sc;
    out->moments = std::move(m);
    out->attempts = attempt;
    out->tip_zone = tipZone(sc);
    out->unpinch = unpinch;
    return true;
  }
  return false;
}

const char* kMetaHeader =
    "family,phase0_gas,nfaces,area1,area2,label_splay,vf_centre,vf_sum3,"
    "thickness,a,b,splay,flare,tip_radius,tip_distance,attempts,small_presence,tip_zone,sample,"
    "neck_thickness,neck_distance,label_unpinch";

void writeSample(long index, const Sample& smp, const Setup& s, std::ostream& mom, std::ostream& nrm,
                 std::ostream& meta, std::ostream* mom5) {
  char buf[64];
  auto put = [&](std::ostream& o, double v, bool last) {
    std::snprintf(buf, sizeof(buf), last ? "%.9g\n" : "%.9g,", v);
    o << buf;
  };
  for (int k = 0; k < 192; ++k) put(mom, smp.features.input[k], k == 191);
  const int nl = smp.features.nlabel;
  for (int k = 0; k < nl; ++k) put(nrm, smp.features.label[k], k == nl - 1);
  if (mom5) for (std::size_t k = 0; k < smp.moments.size(); ++k) put(*mom5, smp.moments[k], k + 1 == smp.moments.size());
  const Truth& t = smp.scene.truth;
  const std::vector<double> m3 = centre3(smp.moments, s.stencil);
  double vfsum = 0.0;
  for (int c = 0; c < 27; ++c) vfsum += m3[7 * c];
  meta << s.families[t.family].name << ',' << int(smp.features.phase0_gas) << ','
       << smp.features.nfaces;
  for (double v : {smp.features.area[0], smp.features.area[1], smp.features.label_splay, m3[7 * 13], vfsum,
                   t.thickness, t.a, t.b, t.splay, t.flare, t.tip_radius, t.tip_distance})
    { std::snprintf(buf, sizeof(buf), ",%.6g", v); meta << buf; }
  meta << ',' << smp.attempts << ',' << smp.features.small_presence << ',' << smp.tip_zone << ',' << index;
  for (double v : {t.neck_thickness, t.neck_distance, smp.unpinch}) { std::snprintf(buf, sizeof(buf), ",%.6g", v); meta << buf; }
  meta << '\n';
}

// Quantile summary of the merged meta file, per family.
void summarize(const std::string& path, const Setup& s) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  struct Acc { long n = 0, two = 0, gas = 0, att = 0, ignore = 0, zone[3] = {0, 0, 0}; std::vector<double> thick, vfc, splay; };
  std::map<std::string, Acc> acc;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<std::string> f;
    for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
    Acc& a = acc[f[0]];
    ++a.n;
    a.two += f[2] == "2";
    a.gas += f[1] == "1";
    a.att += std::stol(f[15]);
    if (std::stod(f[8]) >= 0.0) a.thick.push_back(std::stod(f[8]));
    a.vfc.push_back(std::stod(f[6]));
    if (std::stod(f[5]) >= 0.0) a.splay.push_back(std::stod(f[5]));
    a.ignore += f[16] == "-1";
    const int z = std::stoi(f[17]);
    if (z >= 0) ++a.zone[z];
  }
  auto q = [](std::vector<double> v) {
    if (v.empty()) return std::string("        -");
    std::sort(v.begin(), v.end());
    char b[96];
    auto at = [&](double p) { return v[std::size_t(p * (v.size() - 1) + 0.5)]; };
    std::snprintf(b, sizeof(b), "%7.3g %7.3g %7.3g", at(0.05), at(0.5), at(0.95));
    return std::string(b);
  };
  std::printf("  %-12s %7s %6s %6s %6s %5s | %-23s | %-23s | %-23s | %s\n", "family", "n", "2-face", "ignore",
              "flip", "tries", "thickness 5/50/95%", "centre VF 5/50/95%", "label splay 5/50/95%",
              "tip: centre/stencil/out");
  for (const auto& f : s.families) {
    const Acc& a = acc[f.name];
    if (a.n == 0) continue;
    char zones[64] = "";
    const long nz = a.zone[0] + a.zone[1] + a.zone[2];
    if (nz) std::snprintf(zones, sizeof(zones), "%4.0f%% %4.0f%% %4.0f%%", 100.0 * a.zone[0] / nz,
                          100.0 * a.zone[1] / nz, 100.0 * a.zone[2] / nz);
    std::printf("  %-12s %7ld %5.1f%% %5.1f%% %5.1f%% %5.2f | %s | %s | %s | %s\n", f.name.c_str(), a.n,
                100.0 * a.two / a.n, 100.0 * a.ignore / a.n, 100.0 * a.gas / a.n, double(a.att) / a.n,
                q(a.thick).c_str(), q(a.vfc).c_str(), q(a.splay).c_str(), zones);
  }
}

void mergeShards(const std::string& base, int nranks, bool header) {
  std::ofstream dest(base + ".txt", std::ios::binary | std::ios::trunc);
  if (header) dest << kMetaHeader << '\n';
  for (int r = 0; r < nranks; ++r) {
    const std::string part = base + "_r" + std::to_string(r) + ".txt";
    std::ifstream in(part, std::ios::binary);
    if (in.peek() != std::ifstream::traits_type::eof()) dest << in.rdbuf();
    in.close();
    std::remove(part.c_str());
  }
}

// Moments against Monte Carlo point membership, and the label transform
// against R2P3D_Net's inverse.
int selftest(const Setup& s) {
  std::mt19937_64 eng(12345);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  int failures = 0;
  for (std::size_t f = 0; f < s.families.size(); ++f) {
    double worst_vf = 0.0, worst_c = 0.0, worst_label = 0.0;
    for (int n = 0; n < 20; ++n) {
      Sample smp;
      Setup quiet = s;
      quiet.noise_sigma = 0.0;
      quiet.label_min_gap_fraction = 0.0;   // the label round-trip checks the unconstrained faces
      if (!generate(quiet, int(f), 1000 * f + n, &smp)) { ++failures; continue; }
      // Monte Carlo on the centre cell and one neighbour.
      for (int c : {13, 4}) {
        const int cN = s.stencil == 3 ? c : ((c / 9) + 1) * 25 + ((c / 3) % 3 + 1) * 5 + (c % 3 + 1);
        const Vec3 cc = cellCentre(s.stencil, cN);
        const int M = 200000;
        long hit = 0;
        Vec3 sum = Vec3::Zero();
        for (int k = 0; k < M; ++k) {
          const Vec3 x = cc + Vec3(u(eng), u(eng), u(eng));
          if (smp.scene.liquid(x)) { ++hit; sum += x - cc; }
        }
        const double vf = double(hit) / M;
        const double* o = &smp.moments[7 * cN];
        worst_vf = std::max(worst_vf, std::abs(vf - o[0]));
        if (hit > M / 20 && hit < M - M / 20)
          worst_c = std::max(worst_c, (sum / double(hit) - Vec3(o[1], o[2], o[3])).norm());
      }
      // Undo the canonical frame the way R2P3D_Net does and compare with the
      // grid-frame face normals.
      int d1, d2;
      const std::vector<double> m3 = centre3(smp.moments, s.stencil);
      const Faces faces = centreFaces(smp.scene);
      const Features F = buildFeatures(m3.data(), faces, 0, s.features, &d1, &d2);
      for (int k = 0; k < 2; ++k) {
        double v[3] = {F.label[3 * k], F.label[3 * k + 1], F.label[3 * k + 2]};
        if (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0) continue;
        inverse(d1, d2, v);
        const Vec3 back(v[0], v[1], v[2]);
        double best = 2.0;
        for (int j = 0; j < 2; ++j) {
          if (faces.area[j] <= 0.0) continue;
          const Vec3 want = F.phase0_gas ? faces.normal[j] : Vec3(-faces.normal[j]);
          best = std::min(best, (back - want).norm());
        }
        worst_label = std::max(worst_label, best);
      }
    }
    const bool ok = worst_vf < 5e-3 && worst_c < 1e-2 && worst_label < 1e-9;
    failures += !ok;
    std::printf("  %-16s VF err %.2e  centroid err %.2e  label round-trip %.1e  %s\n", s.families[f].name.c_str(),
                worst_vf, worst_c, worst_label, ok ? "ok" : "FAIL");
  }
  std::printf("self-test %s (Monte Carlo noise ~1e-3)\n", failures ? "FAILED" : "passed");
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (argc < 2) {
    if (rank == 0) std::fprintf(stderr, "usage: r2p_datagen CONFIG [--selftest]\n");
    MPI_Finalize();
    return 2;
  }
  Setup s;
  try {
    s = Setup::fromConfig(Config::load(argv[1]));
  } catch (const std::exception& e) {
    if (rank == 0) std::fprintf(stderr, "config error: %s\n", e.what());
    MPI_Finalize();
    return 2;
  }
  if (rank == 0) {
    std::printf("config %s\n", std::filesystem::absolute(argv[1]).c_str());
    std::printf("  output_dir %s | stencil %d | label.format %s | phase0 rule %s | families %zu\n",
                std::filesystem::absolute(s.output_dir).c_str(), s.stencil,
                s.features.format == LabelFormat::kPresence ? "presence" : "legacy",
                s.features.phase0 == Phase0Rule::kPCA ? "pca" : s.features.phase0 == Phase0Rule::kFilm ? "film" : "vfsum",
                s.families.size());
    std::printf("  splits:");
    for (const auto& sp : s.splits) std::printf(" %s=%ld", sp.name.c_str(), sp.count);
    std::printf(" | viz %s\n", s.viz_per_family > 0
                                   ? (std::to_string(s.viz_per_family) + " per family -> " + s.output_dir + "/" + s.viz_dir).c_str()
                                   : "off");
    std::fflush(stdout);
  }
  // Anything after CONFIG must be a complete, known mode: a typo or a missing
  // argument must never fall through to a full generation run that
  // overwrites the dataset in output_dir.
  const bool selftest_mode = argc == 3 && std::string(argv[2]) == "--selftest";
  const bool viz_mode = argc == 5 && std::string(argv[2]) == "--viz";
  if (argc > 2 && !selftest_mode && !viz_mode) {
    if (rank == 0)
      std::fprintf(stderr, "usage: r2p_datagen CONFIG [--selftest | --viz SPLIT I[,I...]]  (nothing written)\n");
    MPI_Finalize();
    return 2;
  }
  if (argc > 2 && std::string(argv[2]) == "--selftest") {
    const int rc = rank == 0 ? selftest(s) : 0;
    MPI_Finalize();
    return rc;
  }
  if (viz_mode) {
    int rc = 0;
    if (rank == 0) {
      const std::string name = argv[3];
      int id = name == "train" ? 0 : name == "val" ? 1 : name == "test" ? 2 : -1;
      if (id < 0) { std::fprintf(stderr, "--viz: split must be train, val or test\n"); rc = 2; }
      const std::string dir = s.output_dir + "/" + s.viz_dir;
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      std::stringstream list(argv[4]);
      for (std::string tok; rc == 0 && std::getline(list, tok, ',');) {
        const long i = std::stol(tok);
        Sample smp;
        if (!generate(s, s.familyOf(i), s.stream(id, i), &smp)) { std::fprintf(stderr, "sample %ld failed\n", i); continue; }
        const std::string tag = dir + "/" + name + "_" + s.families[smp.scene.truth.family].name + "_" + tok;
        writeViz(tag, smp.scene, smp.moments, s.stencil, s.viz_length);
        std::printf("  %s_*.vtk\n", tag.c_str());
      }
    }
    MPI_Finalize();
    return rc;
  }

  // Indices to visualize: the first viz.per_family samples of each family in
  // the first split (fixed by index alone, like everything else).
  std::set<long> viz;
  if (s.viz_per_family > 0 && !s.splits.empty()) {
    std::vector<int> count(s.families.size(), 0);
    int done = 0;
    for (long i = 0; i < s.splits[0].count && done < int(s.families.size()); ++i) {
      const int f = s.familyOf(i);
      if (count[f] < s.viz_per_family) {
        viz.insert(i);
        if (++count[f] == s.viz_per_family) ++done;
      }
    }
    if (rank == 0) {
      std::error_code ec;
      std::filesystem::create_directories(s.output_dir + "/" + s.viz_dir, ec);
      if (ec) std::fprintf(stderr, "viz: cannot create %s/%s: %s\n", s.output_dir.c_str(), s.viz_dir.c_str(),
                           ec.message().c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  for (std::size_t split = 0; split < s.splits.size(); ++split) {
    const std::string& suffix = s.splits[split].suffix;
    const long total = s.splits[split].count;
    const long begin = total * rank / nranks, end = total * (rank + 1) / nranks;
    const std::string dir = s.output_dir + "/";
    const std::string tag = "_r" + std::to_string(rank) + ".txt";
    std::ofstream mom(dir + "moments" + suffix + tag), nrm(dir + "normals" + suffix + tag),
        meta(dir + "meta" + suffix + tag);
    std::unique_ptr<std::ofstream> mom5;
    if (s.stencil == 5) mom5 = std::make_unique<std::ofstream>(dir + "moments5" + suffix + tag);
    long failed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = begin; i < end; ++i) {
      Sample smp;
      if (!generate(s, s.familyOf(i), s.stream(s.splits[split].id, i), &smp)) { ++failed; continue; }
      writeSample(i, smp, s, mom, nrm, meta, mom5.get());
      if (split == 0 && viz.count(i))
        writeViz(dir + s.viz_dir + "/" + s.families[smp.scene.truth.family].name + "_" + std::to_string(i), smp.scene,
                 smp.moments, s.stencil, s.viz_length);
      if (rank == 0 && (i - begin) % 20000 == 0 && i > begin)
        std::printf("  %s: %ld / %ld on rank 0\n", suffix.empty() ? "train" : suffix.c_str() + 1, i - begin, end - begin);
    }
    mom.close(); nrm.close(); meta.close();
    if (mom5) mom5->close();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    long failed_total = 0;
    double secs_max = 0.0;
    MPI_Reduce(&failed, &failed_total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&secs, &secs_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      for (const std::string base : {"moments", "normals"}) mergeShards(dir + base + suffix, nranks, false);
      mergeShards(dir + "meta" + suffix, nranks, true);
      if (s.stencil == 5) mergeShards(dir + "moments5" + suffix, nranks, false);
      std::printf("[%s] %ld samples in %.1f s on %d rank(s)%s\n", suffix.empty() ? "train" : suffix.c_str() + 1,
                  total - failed_total, secs_max, nranks,
                  failed_total ? (" -- " + std::to_string(failed_total) + " FAILED after max_attempts").c_str() : "");
      summarize(dir + "meta" + suffix + ".txt", s);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
