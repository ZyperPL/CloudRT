#include "weather_entry_view.hpp"
#include "weather_entry.hpp"

#include "imgui.h"

void WeatherEntryView::set_model(std::shared_ptr<WeatherEntry> model) 
{
  this->model = std::move(model);
}

void WeatherEntryView::render_ui() 
{
  assert(model);

  auto list_box_data = [](auto &container, std::string label, std::string format)
  {
    if (ImGui::CollapsingHeader(label.c_str()))
    {
      ImGui::BeginListBox(label.c_str());
      for (size_t i = 0; i < container.size(); ++i)
      {
        ImGui::Text(format.c_str(), container[i]);
      }

      ImGui::EndListBox();
    }
  };

  ImGui::Text("%s", model->get_name().c_str());
  ImGui::Text("Coordinates: %f %f\nHeight: %f", model->get_latitude(), model->get_longitude(), model->get_height());
  list_box_data(model->get_timestamps(), "Timestamps", "%d");
  list_box_data(model->get_lowclouds(), "Low clouds", "%4.1f%%");
  list_box_data(model->get_midclouds(), "Mid-clouds", "%4.1f%%");
  list_box_data(model->get_highclouds(), "High clouds", "%4.1f%%");
  list_box_data(model->get_cloudwater(), "Cloud water", "%5.1f g");
  list_box_data(model->get_cloudice(), "Cloud ice", "%5.1f g");
  list_box_data(model->get_surfaceairpressure(), "Surface air pressure", "%5f hPa");
}
