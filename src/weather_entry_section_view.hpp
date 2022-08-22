#pragma once

#include <memory>

#include "weather_entry.hpp"
#include "weather_entry_section.hpp"

class WeatherEntrySectionView
{
public:
  void render_ui(const WeatherEntrySection &section);
  void render_range(std::shared_ptr<WeatherEntry> entry, ssize_t index, ssize_t range);
};
