#include "datetime_view.hpp"

#include <ctime>

DateTimeView::DateTimeView(DateTimeViewObserver *observer)
    : observer{observer} {}

void DateTimeView::date_ui(std::shared_ptr<WeatherEntry> entry,
                           ssize_t &selected_index) {

  assert(selected_index >= 0);

  const auto &timestamp = entry->get_timestamps()[selected_index];

  if (ImGui::Button("◀ Previous day", ImVec2(200, 32))) {
    observer->onPreviousDayButtonPressed();

    return;
  }

  ImGui::SameLine();

  auto time = std::gmtime(&timestamp);
  ImGui::Text(" %02d:%02d:%02d %d.%02d.%04d ", time->tm_hour, time->tm_min,
              time->tm_sec, time->tm_mday, time->tm_mon + 1,
              time->tm_year + 1900);

  ImGui::SameLine();

  if (ImGui::Button("Next day ▶", ImVec2(200, 32))) {
    observer->onNextDayButtonPressed();
    return;
  }

  ImGui::SameLine();

  if (ImGui::Button("Today", ImVec2(80, 32))) {
    observer->onTodayButtonPressed();
    return;
  }
}

void DateTimeView::time_ui([[maybe_unused]] std::shared_ptr<WeatherEntry> view,
                           ssize_t &selected_index) {
  const int previous_day_offset = ((selected_index % 24 + 24) - 2) % 24;
  int day_offset = previous_day_offset;
  // TODO: cap min and max values based on timestamps count
  ImGui::SliderInt("Hour", &day_offset, 0, 23);

  if (day_offset != previous_day_offset)
    observer->onTimeSliderChanged(day_offset - previous_day_offset);
}
