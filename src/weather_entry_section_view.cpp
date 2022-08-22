#include "weather_entry_section_view.hpp"

#include "imgui.h"

void WeatherEntrySectionView::render_ui(const WeatherEntrySection &section) {

  ImGui::Text("%f;%f (%f a.s.l)", section.latitude, section.longitude,
              section.height);
  ImGui::Text("Low clouds: %2.1f %%", section.low_clouds);
  ImGui::Text("Mid clouds: %2.1f %%", section.mid_clouds);
  ImGui::Text("High clouds: %2.1f %%", section.high_clouds);
  ImGui::Text("Cloud water: %2.1f g", section.cloud_water);
  ImGui::Text("Cloud ice: %2.1f g", section.cloud_ice);
}

void WeatherEntrySectionView::render_range(std::shared_ptr<WeatherEntry> entry,
                                           ssize_t index, ssize_t range) {
  if (!entry || index < 0)
    return;

  std::vector<float> low_clouds_values;
  std::vector<float> mid_clouds_values;
  std::vector<float> high_clouds_values;
  std::vector<float> water_values;
  std::vector<float> ice_values;
  for (ssize_t i = index - range / 2; i < index + range / 2; ++i) {
    auto section = entry->section(i);
    if (!section)
      continue;

    low_clouds_values.push_back(static_cast<float>(section->low_clouds));
    mid_clouds_values.push_back(static_cast<float>(section->mid_clouds));
    high_clouds_values.push_back(static_cast<float>(section->high_clouds));
    water_values.push_back(static_cast<float>(section->cloud_water));
    ice_values.push_back(static_cast<float>(section->cloud_ice));
  }

  auto plot_floats = [](std::string text, std::vector<float> &values, float scale_min, float scale_max) {
    ImGui::PushID(text.c_str());
    const auto avail = ImGui::GetContentRegionAvail();
    ImGui::Text("%s", text.c_str());
    ImGui::PlotHistogram("##plotfloatshistogram", values.data(), values.size(),
                         0, NULL, scale_min, scale_max, ImVec2(avail.x - 20, 60));
    ImGui::PopID();
  };

  plot_floats("Low clouds", low_clouds_values, 0.0f, 100.0f);
  plot_floats("Mid clouds", mid_clouds_values, 0.0f, 100.0f);
  plot_floats("High clouds", high_clouds_values, 0.0f, 100.0f);
  plot_floats("Cloud water", water_values, 0.0f, 100.0f);
  plot_floats("Cloud ice", ice_values, 0.0f, 100.0f);
}
