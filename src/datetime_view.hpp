#pragma once

#include "imgui.h"

#include "weather_entry.hpp"

class DateTimeView {
public:
  void set_entry(std::shared_ptr<WeatherEntry> entry);
  void date_ui(ssize_t &selected_index);
  void time_ui(ssize_t &selected_index);

private:
  std::shared_ptr<WeatherEntry> entry;
};
