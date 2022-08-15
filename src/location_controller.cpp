#include "location_controller.hpp"

#include "http.hpp"

void LocationController::execute() {
  const bool has_no_selected_location =
      selected_location < 0 &&
      selected_location < static_cast<ssize_t>(locations_names.size());

  if (has_no_selected_location) {

    if (auto input_location_optional = view.get_location_input()) {
      assert(input_location_optional.has_value());
      std::string input_location_string = input_location_optional.value();
      HTTP::QueryParameters query_parameters;
      query_parameters.add("query", input_location_string);

      // TODO: read from config
      auto data_obj = HTTP::get("localhost:5000/location", query_parameters);

      if (data_obj) {
        // TODO: handle API error
        nlohmann::json j = nlohmann::json::parse(data_obj->get()->data);
        locations_names.clear();
        locations_json.clear();
        if (j.contains("count") && j.contains("results")) {
          const size_t count = j["count"];
          const auto result = j["results"];
          const size_t results_count =
              std::min(result.size(), std::min(static_cast<size_t>(7), count));
          for (size_t i = 0; i < results_count; i++) {
            auto res = result[i];
            std::string name = view.construct_name_string(result[i]);
            locations_names.push_back(name);
            locations_json.push_back(result[i]);
          }
        } else {
          printf("%s\n", j.dump(2).c_str());
        }
      }
      // TODO: handle connection error
    }

    ssize_t clicked_locations_list_index = -1;
    view.locations_list(locations_names, selected_location,
                        clicked_locations_list_index);

    assert(clicked_locations_list_index >= -1);
    assert(clicked_locations_list_index < (ssize_t)(locations_names.size()));

    if (clicked_locations_list_index >= 0) {
      selected_location = clicked_locations_list_index;

      assert(selected_location < (ssize_t)(locations_json.size()));
      const auto &selected_json = locations_json[selected_location];

      HTTP::QueryParameters query_parameters;
      std::string selected_name{};

      if (selected_json.contains("name") && selected_json.contains("lat") &&
          selected_json.contains("lon")) {

        if (selected_json["name"].is_string())
          selected_name = selected_json["name"].get<std::string>();

        printf("Selected name: %s index: %zu\n", selected_name.c_str(),
               selected_location);

        if (selected_json["lat"].is_number_float()) {
          const auto &lat = selected_json["lat"];
          query_parameters.add("lat", std::to_string(lat.get<float>()));
        }
        if (selected_json["lon"].is_number_float()) {
          const auto &lon = selected_json["lon"];
          query_parameters.add("lon", std::to_string(lon.get<float>()));
        }
      }

      auto data_object = HTTP::get("localhost:5000/weather", query_parameters);

      if (data_object) {
        // TODO: handle API error
        nlohmann::json json = nlohmann::json::parse(data_object->get()->data);
        entry = std::make_shared<WeatherEntry>(
            json, locations_json[selected_location]["name"].get<std::string>());
      }
      // TODO: handle connection error
    }
  } else {
    ImGui::PushID(selected_location);

    // button to allow entering location again
    if (view.location_button(locations_names[selected_location]))
      selected_location = -1;

    ImGui::PopID();
  }
}
