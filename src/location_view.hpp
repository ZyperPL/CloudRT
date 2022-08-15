#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "imgui.h"
#include <nlohmann/json.hpp>

#include "weather_entry.hpp"

class LocationView {
public:
  void locations_list(const std::vector<std::string> &names,
                      const ssize_t selected_index, ssize_t &clicked_index);

  std::string construct_name_string(nlohmann::json);
  std::optional<std::string> get_location_input();
  bool location_button(std::string location_name);

  static const size_t MAX_INPUT_STRING_SIZE;

private:
  std::string input_location_string;
  std::string previous_input_location_string{" "};
};
