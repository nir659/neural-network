#include "csv.hpp"
#include <fstream>
#include <sstream>
#include <vector>

tensor load_csv(const std::string &path) {
  std::ifstream file(path);
  std::string line, cell;
  std::vector<float> float_data;
  size_t row = 0;
  size_t cols = 0;

  bool first_line = true;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    if (first_line && !line.empty() && !std::isdigit(static_cast<unsigned char>(line[0])) && line[0] != '-') {
      first_line = false;
      continue;
    }
    first_line = false;

    std::stringstream ss(line);
    size_t current_cols = 0;

    while (std::getline(ss, cell, ',')) {
      float_data.push_back(std::stof(cell));
      current_cols++;
    }
    if (row == 0)
      cols = current_cols;
    row++;
  }

  tensor result({row, cols});
  result.data = std::move(float_data);
  return result;
}