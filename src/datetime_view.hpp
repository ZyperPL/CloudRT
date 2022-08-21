#pragma once

#include "imgui.h"

#include "datetime_view_observer.hpp"
#include "weather_entry.hpp"

class DateTimeView {
public:
  DateTimeView(DateTimeViewObserver *observer);
  void date_ui(std::shared_ptr<WeatherEntry>, ssize_t &selected_index);
  void time_ui(std::shared_ptr<WeatherEntry>, ssize_t &selected_index);

private:
  DateTimeViewObserver *observer{nullptr};
};
