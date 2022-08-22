#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "datetime_controller.hpp"
#include "location_controller.hpp"
#include "weather_entry_view.hpp"
#include "weather_mapper.hpp"

struct GLFWwindow;
class Texture;

class WindowMain {
public:
  WindowMain();
  ~WindowMain();
  void render();
  bool is_open();

private:
  GLFWwindow *handle{nullptr};
  std::unique_ptr<Texture> render_texture;
  std::unique_ptr<Texture> clouds_texture;

  WeatherEntryView weather_entry_view;
  DateTimeController date_time_controller;
  LocationController location_controller;
  WeatherMapper mapper;

  RenderParameters render_parameters;
  CloudsRenderParameters clouds_texture_parameters;
};
