#pragma once

#include <memory>

#include "cloud_noise.hpp"
#include "render.hpp"
#include "weather_entry.hpp"
#include "weather_entry_section.hpp"

class WeatherMapper {
public:
  void set_entry(std::shared_ptr<WeatherEntry> entry) {
    this->entry = std::move(entry);
  }

  bool has_entry() { return !!entry; }

  void calculate(size_t index);

  std::shared_ptr<RenderParameters> get_render_parameters() {
    return render_parameters;
  }

  std::shared_ptr<CloudsRenderParameters> get_clouds_texture_parameters() {
    return clouds_texture_parameters;
  }

private:
  std::shared_ptr<WeatherEntry> entry;
  std::shared_ptr<RenderParameters> render_parameters;
  std::shared_ptr<CloudsRenderParameters> clouds_texture_parameters;

  void calculate_render_parameters(WeatherEntrySection&);
  void calculate_clouds_texture_parameters(WeatherEntrySection&);
};
