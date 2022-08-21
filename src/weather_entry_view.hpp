#pragma once

#include <memory>

class WeatherEntry;

class WeatherEntryView {
public:
  void render_ui(std::shared_ptr<WeatherEntry>);

private:
};
