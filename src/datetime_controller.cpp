#include "datetime_controller.hpp"

#include "imgui.h"

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

  this->view.set_entry(entry);
  this->view.date_ui(selected_index);
  this->view.time_ui(selected_index);
}
