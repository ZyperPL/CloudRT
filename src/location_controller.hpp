#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "imgui.h"
#include <nlohmann/json.hpp>

#include "location_view.hpp"
#include "weather_entry.hpp"
#include "location_view_observer.hpp"

class LocationController : public LocationViewObserver {
public:
  LocationController();
  void execute();

  std::shared_ptr<WeatherEntry> get_entry() { return entry; }
  bool has_entry() { return !!entry; }

  void onLocationSelected(size_t);
  void onLocationInputTextChanged(const std::string&);
  void onButtonPressed();
private:
  LocationView view;
  std::vector<std::string> locations_names;
  std::vector<nlohmann::json> locations_json;
  ssize_t selected_location{-1};
  std::shared_ptr<WeatherEntry> entry;
};
