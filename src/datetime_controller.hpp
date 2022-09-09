#pragma once

#include <memory>
#include <vector>

#include "datetime_view.hpp"
#include "datetime_view_observer.hpp"
#include "weather_entry.hpp"

class DateTimeController : public DateTimeViewObserver {
public:
  DateTimeController();

  void set_entry(std::shared_ptr<WeatherEntry> entry);
  void execute();

  ssize_t get_index() const { return selected_index; }

  void onNextDayButtonPressed();
  void onPreviousDayButtonPressed();
  void onTodayButtonPressed();
  void onTimeSliderChanged(size_t);

private:
  DateTimeView view;

  std::shared_ptr<WeatherEntry> entry;
  ssize_t selected_index{-1};
};
