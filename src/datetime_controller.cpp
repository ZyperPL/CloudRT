#include "datetime_controller.hpp"

#include "imgui.h"

DateTimeController::DateTimeController() : view{this} {}

void DateTimeController::set_entry(std::shared_ptr<WeatherEntry> entry) {
  this->entry = std::move(entry);
}

void DateTimeController::execute() {
  if (!entry) {
    ImGui::Text("Location entry not selected!");
    return;
  }

  if (selected_index < 0)
    selected_index = 0;

  if (selected_index >= static_cast<ssize_t>(entry->count()))
    selected_index = entry->count() - 1;

  this->view.date_ui(entry, selected_index);
  this->view.time_ui(entry, selected_index);
}

void DateTimeController::onNextDayButtonPressed() {

  const ssize_t max_index = static_cast<ssize_t>(entry->count());
  if (selected_index < max_index - 24)
    selected_index += 24;
}

void DateTimeController::onPreviousDayButtonPressed() {
  if (selected_index >= 24)
    selected_index -= 24;
}
void DateTimeController::onTodayButtonPressed() { 
  //TODO: make setting "today" more precise
  selected_index = 4 + 24 * 4; 
}
void DateTimeController::onTimeSliderChanged(size_t offset) {
  selected_index += offset;
}
