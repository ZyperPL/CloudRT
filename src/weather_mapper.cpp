#include "weather_mapper.hpp"

void WeatherMapper::calculate(size_t index) {
  if (!entry)
    return;

  if (auto entry_section = entry->section(index)) {
    calculate_render_parameters(entry_section.value());
    calculate_clouds_texture_parameters(entry_section.value());
    assert(render_parameters);
    assert(clouds_texture_parameters);
  }
}

void WeatherMapper::calculate_render_parameters(WeatherEntrySection &section) {
  render_parameters = std::make_shared<RenderParameters>();
  auto &p = *render_parameters;

  p.camera_position = glm::vec3(0.0f, 1.0f, -1.0f);
  p.camera_position.y += glm::mix(0.0f, 10.0f, section.height / 1000.0f);

  float water = section.cloud_water + section.cloud_ice * 0.1f;
  const float MAX_WATER = 400.0f;
  p.camera_direction = glm::vec3(0.0f, 0.3f, 0.7f);
  p.density = glm::clamp(1.0f - glm::mix(0.0f, 0.8f, (water / MAX_WATER)), 0.2f, 1.0f);
  p.light_direction = glm::vec3(0.6f, 0.65f, -0.8f);
}

void WeatherMapper::calculate_clouds_texture_parameters(
    WeatherEntrySection &section) {
  clouds_texture_parameters = std::make_shared<CloudsRenderParameters>();
  auto &p = *clouds_texture_parameters;

  p.frequency = 2.0f;
  p.octaves = 3.0f;

  const double &lc = section.low_clouds;
  p.low_cut_l = 0.15 - glm::smoothstep(0.0, 0.15, lc / 100.0) * 0.15;
  p.high_cut_l = 1.0 - glm::smoothstep(0.5, 1.000, lc / 100.0) * 0.5;

  const double &mc = section.mid_clouds;
  p.low_cut_m = 0.15 - glm::smoothstep(0.05, 0.15, mc / 100.0) * 0.05;
  p.high_cut_m = 1.0 - glm::smoothstep(0.5, 1.000, mc / 100.0) * 0.5;

  const double &hc = section.mid_clouds;
  p.low_cut_h = 0.15 - glm::smoothstep(0.05, 0.15, hc / 100.0) * 0.05;
  p.high_cut_h = 1.0 - glm::smoothstep(0.5, 1.000, hc / 100.0) * 0.5;
}
