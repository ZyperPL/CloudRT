#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "imgui.h"
#include <nlohmann/json.hpp>

#include "weather_entry.hpp"

#include "location_view_observer.hpp"

class LocationView {
public:
  LocationView(LocationViewObserver *observer);
  void locations_list(const std::vector<std::string> &names,
                      const ssize_t selected_index);

  std::string construct_name_string(nlohmann::json);
  void location_input();
  void location_button(std::string location_name);

  static const size_t MAX_INPUT_STRING_SIZE;

private:
  LocationViewObserver *observer;
  std::string input_location_string;
  std::string previous_input_location_string{" "};
};
