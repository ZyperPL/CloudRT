#include "datetime_view.hpp"

#include <ctime>

void DateTimeView::set_entry(std::shared_ptr<WeatherEntry> entry) {
  this->entry = std::move(entry);
}

void DateTimeView::date_ui(ssize_t &selected_index) {
  const ssize_t max_index = static_cast<ssize_t>(entry->count());

  assert(selected_index >= 0);
  assert(selected_index < max_index);

  const auto &timestamp = entry->get_timestamps()[selected_index];

  if (ImGui::Button("<", ImVec2(24, 24)))
  {
    if (selected_index >= 24)
      selected_index -= 24;
    return;
  }

  ImGui::SameLine();

  auto time = std::gmtime(&timestamp);
  ImGui::Text("%02d:%02d:%02d %d.%02d.%04d\n", time->tm_hour, time->tm_min,
              time->tm_sec, time->tm_mday, time->tm_mon + 1,
              time->tm_year + 1900);

  ImGui::SameLine();

  if (ImGui::Button(">", ImVec2(24, 24)))
  {
    if (selected_index < max_index - 24)
      selected_index += 24;
    return;
  }
}

void DateTimeView::time_ui(ssize_t &selected_index) {
  const int previous_day_offset = ((selected_index % 24 + 24) - 2) % 24;
  int day_offset = previous_day_offset;
  //TODO: cap min and max values based on timestamps count
  ImGui::SliderInt("Hour", &day_offset, 0, 23);

  if (day_offset != previous_day_offset)
    selected_index += day_offset - previous_day_offset;
}
