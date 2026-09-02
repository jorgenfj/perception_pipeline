// Builds recording/include/ros_schemas.hpp out of real ROS 2 .msg files.
//
// A ros2msg schema is the type's own .msg text, then each type it depends on
// behind an 80-character rule and a `MSG: <pkg>/<Type>` line -- concatenation,
// so no ROS library is linked and ROS is only a data dependency.
//
// The output is committed, because the recorder must build with no ROS present.
// That makes staleness possible, which is what --check exists for.
//
// Definitions are passed through untouched, comments included: rosbag2 keeps
// them and Foxglove shows them.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// 80, per the MCAP well-known-profile registry. The same rule is embedded in
// librosbag2_storage_mcap.so, and in the schemas this project hand-wrote first.
constexpr const char* kRule =
    "================================================================================";

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("ros_schema_gen: " + what);
}

// Everything CDR encodes without a message definition behind it.
bool is_primitive(std::string_view type) {
  static constexpr std::string_view kPrimitives[] = {
      "bool",   "byte",   "char",   "float32", "float64", "int8",   "uint8",
      "int16",  "uint16", "int32",  "uint32",  "int64",   "uint64", "string",
      "wstring"};
  return std::find(std::begin(kPrimitives), std::end(kPrimitives), type) !=
         std::end(kPrimitives);
}

// "geometry_msgs/msg/Vector3" and "geometry_msgs/Vector3" name one type. The
// .msg files use the short form; a types.txt line uses the long one.
std::string normalise(std::string_view name) {
  const std::size_t first = name.find('/');
  if (first == std::string_view::npos) return std::string(name);
  const std::size_t second = name.find('/', first + 1);
  if (second == std::string_view::npos) return std::string(name);
  return std::string(name.substr(0, first)) + "/" + std::string(name.substr(second + 1));
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot read '" + path.string() + "'");
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

// <search>/<pkg>/msg/<Type>.msg, in the order the search paths were given, so a
// definition in this repo shadows nothing and is found the same way as one in
// /opt/ros.
std::string find_definition(const std::string& type,
                            const std::vector<std::filesystem::path>& search) {
  const std::size_t slash = type.find('/');
  if (slash == std::string::npos) fail("'" + type + "' is not <package>/<Type>");
  const std::string package = type.substr(0, slash);
  const std::string name = type.substr(slash + 1);

  for (const std::filesystem::path& root : search) {
    const std::filesystem::path candidate = root / package / "msg" / (name + ".msg");
    if (std::filesystem::exists(candidate)) return read_file(candidate);
  }

  std::string tried;
  for (const std::filesystem::path& root : search) tried += "\n    " + root.string();
  fail("no definition for '" + type + "'; searched:" + tried);
}

// The field types this definition references. `package` is its own, because a
// .msg may name a sibling unqualified: TwistWithCovarianceStamped says
// `TwistWithCovariance twist`, and Twist says `Vector3 linear`.
std::vector<std::string> dependencies_of(const std::string& definition,
                                         const std::string& package) {
  std::vector<std::string> out;
  std::istringstream lines(definition);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);

    std::istringstream tokens(line);
    std::string type;
    std::string field;
    if (!(tokens >> type)) continue;

    // Arrays of any flavour -- [], [9], [<=4] -- are the element type.
    const std::size_t bracket = type.find('[');
    if (bracket != std::string::npos) type.resize(bracket);
    if (type.empty() || is_primitive(type)) continue;

    // A constant, not a field: `uint8 ULTRASOUND=0` in sensor_msgs/Range.
    if (tokens >> field && field.find('=') != std::string::npos) continue;

    // ROS 1 wrote a bare `Header`; ROS 2 spells it out. Anything else unqualified
    // is a sibling in this message's own package.
    std::string full;
    if (type == "Header") {
      full = "std_msgs/Header";
    } else if (type.find('/') == std::string::npos) {
      full = package + "/" + type;
    } else {
      full = normalise(type);
    }
    if (std::find(out.begin(), out.end(), full) == out.end()) out.push_back(full);
  }
  return out;
}

// Depth first in field order, each type once however many times it is reached.
// TwistWithCovarianceStamped reaches Vector3 twice, and a schema that defines a
// type twice is one some readers reject.
void collect(const std::string& type, const std::vector<std::filesystem::path>& search,
             std::vector<std::string>& order, std::vector<std::string>& seen) {
  if (std::find(seen.begin(), seen.end(), type) != seen.end()) return;
  seen.push_back(type);
  order.push_back(type);
  const std::string package = type.substr(0, type.find('/'));
  for (const std::string& dep : dependencies_of(find_definition(type, search), package)) {
    collect(dep, search, order, seen);
  }
}

std::string build_schema(const std::string& type,
                         const std::vector<std::filesystem::path>& search) {
  std::vector<std::string> order;
  std::vector<std::string> seen;
  collect(normalise(type), search, order, seen);

  std::string out = find_definition(order.front(), search);
  for (std::size_t i = 1; i < order.size(); ++i) {
    if (!out.empty() && out.back() != '\n') out += '\n';
    out += kRule;
    out += "\nMSG: " + order[i] + "\n";
    out += find_definition(order[i], search);
  }
  return out;
}

// "sensor_msgs/msg/Imu" -> "kImu". The package is dropped because it reads
// better and because two ROS packages defining one type name is a collision
// worth being told about rather than mangling around.
std::string constant_name(const std::string& type) {
  const std::size_t slash = type.rfind('/');
  return "k" + (slash == std::string::npos ? type : type.substr(slash + 1));
}

std::vector<std::string> read_types(const std::filesystem::path& path) {
  std::vector<std::string> out;
  std::istringstream lines(read_file(path));
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    std::istringstream tokens(line);
    std::string type;
    if (tokens >> type) out.push_back(type);
  }
  if (out.empty()) fail("'" + path.string() + "' lists no types");
  return out;
}

// No timestamp and no host path in the banner: the output has to be identical
// on every machine or --check would fail on all but the one that generated it.
std::string build_header(const std::vector<std::string>& types,
                         const std::vector<std::filesystem::path>& search) {
  std::string out =
      "#pragma once\n"
      "\n"
      "// GENERATED by tools/ros_schemas from installed ROS 2 .msg files.\n"
      "// Do not edit: regenerate with\n"
      "//\n"
      "//   cmake -S . -B build-ros -DPERCEPTION_WITH_ROS=ON\n"
      "//   cmake --build build-ros --target ros_schemas\n"
      "//\n"
      "// The types are listed in tools/ros_schemas/types.txt. This file is committed\n"
      "// because the recorder must build, and its recordings must be readable, on a\n"
      "// machine with no ROS installed -- see recording/src/mcap_recorder.cpp.\n"
      "\n"
      "#include <string_view>\n"
      "\n"
      "namespace perception::ros_msg::schema {\n";

  std::vector<std::string> used;
  for (const std::string& type : types) {
    const std::string name = constant_name(type);
    if (std::find(used.begin(), used.end(), name) != used.end()) {
      fail("two types would both generate '" + name + "'; one of them needs renaming");
    }
    used.push_back(name);

    const std::string schema = build_schema(type, search);
    if (schema.find(")MSGDEF\"") != std::string::npos) {
      fail("'" + type + "' contains the raw-string delimiter; pick another one");
    }

    out += "\n// " + type + "\n";
    out += "inline constexpr std::string_view " + name + "Type = \"" + type + "\";\n";
    out += "inline constexpr std::string_view " + name + " = R\"MSGDEF(" + schema + ")MSGDEF\";\n";
  }

  out += "\n}  // namespace perception::ros_msg::schema\n";
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<std::filesystem::path> search;
    std::filesystem::path types_path;
    std::filesystem::path out_path;
    bool check_only = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&]() -> std::string {
        if (i + 1 >= argc) fail(arg + " needs a value");
        return argv[++i];
      };
      if (arg == "--search") search.emplace_back(value());
      else if (arg == "--types") types_path = value();
      else if (arg == "--out") out_path = value();
      else if (arg == "--check") check_only = true;
      else fail("unknown argument '" + arg + "'");
    }

    if (search.empty()) fail("--search is required (e.g. /opt/ros/humble/share)");
    if (types_path.empty()) fail("--types is required");
    if (out_path.empty()) fail("--out is required");

    const std::string generated = build_header(read_types(types_path), search);

    if (check_only) {
      if (!std::filesystem::exists(out_path)) {
        fail("'" + out_path.string() + "' does not exist; generate it and commit it");
      }
      if (read_file(out_path) != generated) {
        fail("'" + out_path.string() +
             "' is stale.\n"
             "     Regenerate it and commit the result:\n"
             "       cmake --build <build> --target ros_schemas");
      }
      std::printf("ros_schema_gen: %s is up to date\n", out_path.string().c_str());
      return 0;
    }

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot write '" + out_path.string() + "'");
    out << generated;
    out.close();
    std::printf("ros_schema_gen: wrote %s (%zu bytes)\n", out_path.string().c_str(),
                generated.size());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
