// Config file for the R2P-Net data generator.
//
// Plain text, one "key = value" per line, '#' starts a comment. Lines before
// the first [section] are global settings; every [section] after that is one
// sample family (see presets.cfg for the full list of keys).
//
// A distribution value is one of
//   fixed V | uniform A B | loguniform A B | normal M S | choice V1 V2 ...
// optionally prefixed with "signed" (random sign on the drawn value, e.g.
// "signed loguniform 0.01 0.25"), and a bare number means "fixed". "inf" is
// accepted wherever a number is.
//
// Every key a family or the global section does not use is reported as an
// error, so a typo cannot silently fall back to a default.

#ifndef EXAMPLES_R2P_DATAGEN_CONFIG_H_
#define EXAMPLES_R2P_DATAGEN_CONFIG_H_

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace r2pgen {

inline double parseNumber(const std::string& s) {
  if (s == "inf" || s == "+inf") return std::numeric_limits<double>::infinity();
  if (s == "-inf") return -std::numeric_limits<double>::infinity();
  std::size_t used = 0;
  const double v = std::stod(s, &used);
  if (used != s.size()) throw std::runtime_error("not a number: '" + s + "'");
  return v;
}

struct Dist {
  enum class Kind { kFixed, kUniform, kLogUniform, kNormal, kChoice };
  Kind kind = Kind::kFixed;
  std::vector<double> p{0.0};
  std::string text = "0";
  bool random_sign = false;

  static Dist parse(const std::string& text) {
    std::stringstream ss(text);
    std::string head;
    ss >> head;
    if (head == "signed") {
      std::string rest;
      std::getline(ss, rest);
      Dist d = parse(rest.substr(rest.find_first_not_of(' ')));
      d.text = text;
      d.random_sign = true;
      return d;
    }
    std::vector<double> args;
    for (std::string tok; ss >> tok;) args.push_back(parseNumber(tok));
    Dist d;
    d.text = text;
    auto need = [&](std::size_t n) {
      if (args.size() != n) throw std::runtime_error("'" + text + "': expected " + std::to_string(n) + " values");
    };
    if (head == "fixed") { need(1); d.kind = Kind::kFixed; }
    else if (head == "uniform") { need(2); d.kind = Kind::kUniform; }
    else if (head == "loguniform") {
      need(2);
      if (!(args[0] > 0.0 && args[1] >= args[0])) throw std::runtime_error("'" + text + "': needs 0 < A <= B");
      d.kind = Kind::kLogUniform;
    }
    else if (head == "normal") { need(2); d.kind = Kind::kNormal; }
    else if (head == "choice") {
      if (args.empty()) throw std::runtime_error("'" + text + "': choice needs values");
      d.kind = Kind::kChoice;
    }
    else { args = {parseNumber(head)}; d.kind = Kind::kFixed; }
    d.p = args;
    return d;
  }

  template <class Engine>
  double operator()(Engine& eng) const {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (random_sign) {
      Dist d = *this;
      d.random_sign = false;
      const double v = d(eng);
      return u(eng) < 0.5 ? -v : v;
    }
    switch (kind) {
      case Kind::kFixed: return p[0];
      case Kind::kUniform: return p[0] + (p[1] - p[0]) * u(eng);
      case Kind::kLogUniform: return std::exp(std::log(p[0]) + (std::log(p[1]) - std::log(p[0])) * u(eng));
      case Kind::kNormal: return p[0] + p[1] * std::normal_distribution<double>(0.0, 1.0)(eng);
      case Kind::kChoice: return p[std::min(p.size() - 1, std::size_t(u(eng) * p.size()))];
    }
    return p[0];
  }
};

// One [section]: key/value pairs, with every accessed key recorded so unused
// (misspelled) keys can be reported.
class Section {
 public:
  std::string name;
  int line = 0;

  void set(const std::string& key, const std::string& value) { kv_[key] = value; }
  bool has(const std::string& key) const { return kv_.count(key) > 0; }

  std::string str(const std::string& key, const std::string& fallback) const {
    used_.insert(key);
    auto it = kv_.find(key);
    return it == kv_.end() ? fallback : it->second;
  }
  double num(const std::string& key, double fallback) const {
    const std::string s = str(key, "");
    return s.empty() ? fallback : parseNumber(s);
  }
  Dist dist(const std::string& key, const std::string& fallback) const {
    const std::string s = str(key, fallback);
    try {
      return Dist::parse(s);
    } catch (const std::exception& e) {
      throw std::runtime_error("[" + name + "] " + key + ": " + e.what());
    }
  }
  std::vector<std::string> unused() const {
    std::vector<std::string> out;
    for (const auto& kv : kv_) if (!used_.count(kv.first)) out.push_back(kv.first);
    return out;
  }

 private:
  std::map<std::string, std::string> kv_;
  mutable std::set<std::string> used_;
};

struct Config {
  Section global;
  std::vector<Section> families;

  static Config load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config " + path);
    Config cfg;
    cfg.global.name = "global";
    Section* cur = &cfg.global;
    std::string line;
    for (int n = 1; std::getline(in, line); ++n) {
      const auto hash = line.find('#');
      if (hash != std::string::npos) line.erase(hash);
      auto trim = [](std::string s) {
        const auto b = s.find_first_not_of(" \t\r");
        const auto e = s.find_last_not_of(" \t\r");
        return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
      };
      line = trim(line);
      if (line.empty()) continue;
      if (line.front() == '[') {
        if (line.back() != ']') throw std::runtime_error(path + ":" + std::to_string(n) + ": bad section header");
        cfg.families.emplace_back();
        cur = &cfg.families.back();
        cur->name = trim(line.substr(1, line.size() - 2));
        cur->line = n;
        continue;
      }
      const auto eq = line.find('=');
      if (eq == std::string::npos) throw std::runtime_error(path + ":" + std::to_string(n) + ": expected key = value");
      cur->set(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
    }
    return cfg;
  }

  // Throws if any section has a key nothing read.
  void checkAllUsed() const {
    std::string msg;
    auto check = [&](const Section& s) {
      for (const auto& k : s.unused()) msg += "  [" + s.name + "] " + k + "\n";
    };
    check(global);
    for (const auto& f : families) check(f);
    if (!msg.empty()) throw std::runtime_error("unknown config keys (typo?):\n" + msg);
  }
};

}  // namespace r2pgen

#endif  // EXAMPLES_R2P_DATAGEN_CONFIG_H_
