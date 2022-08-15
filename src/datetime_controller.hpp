#pragma once

#include <memory>
#include <vector>

#include "weather_entry.hpp"
#include "datetime_view.hpp"

class DateTimeController
{
  public:
    void set_entry(std::shared_ptr<WeatherEntry> entry);
    void execute();

    ssize_t get_index() const { return selected_index; }
  private:
    DateTimeView view;

    std::shared_ptr<WeatherEntry> entry;
    ssize_t selected_index { 3 };
};
