#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "imgui.h"
#include <nlohmann/json.hpp>

#include "location_view.hpp"
#include "weather_entry.hpp"

class LocationController {
public:
  void execute();

  std::shared_ptr<WeatherEntry> get_entry() { return entry; }
  bool has_entry() { return !!entry; }

private:
  LocationView view;
  std::vector<std::string> locations_names;
  std::vector<nlohmann::json> locations_json;
  ssize_t selected_location{-1};
  std::shared_ptr<WeatherEntry> entry;
};
