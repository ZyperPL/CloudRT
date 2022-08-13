#pragma once

#include <memory>

class WeatherEntry;

namespace View {
class WeatherEntryView {
public:
  void set_model(std::shared_ptr<WeatherEntry>);
  void render_ui();

private:
  std::shared_ptr<WeatherEntry> model;
};
}; // namespace View
