#include "location_view.hpp"

const size_t LocationView::MAX_INPUT_STRING_SIZE = 2048;

void LocationView::locations_list(const std::vector<std::string> &names,
                                  const ssize_t selected_index,
                                  ssize_t &clicked_index) {
  if (ImGui::BeginListBox("##locationslistbox")) {
    for (size_t name_idx = 0; name_idx < names.size(); ++name_idx) {
      ImGui::PushID(name_idx);
      const bool is_selected =
          (static_cast<size_t>(selected_index) == name_idx);
      if (ImGui::Selectable(names[name_idx].c_str(), is_selected)) {
        clicked_index = name_idx;
      }
      ImGui::PopID();
    }
  }
  ImGui::EndListBox();
}

std::string LocationView::construct_name_string(nlohmann::json json_object) {
  std::string name{};
  if (json_object.contains("name")) {
    name += json_object["name"];
    if (json_object.contains("admin1")) {
      name += ", ";
      name += json_object["admin1"];
    }
    if (json_object.contains("country")) {
      name += ", ";
      name += json_object["country"];
    }
  }

  return name;
}

std::optional<std::string> LocationView::get_location_input() {
  const char *LOCATION_HINT = "Type location name";
  input_location_string.resize(MAX_INPUT_STRING_SIZE, '\0');
  ImGui::InputTextWithHint("Location", LOCATION_HINT,
                           input_location_string.data(),
                           MAX_INPUT_STRING_SIZE - 1);
  input_location_string.resize(input_location_string.find('\0'));

  const bool has_new_string_input =
      previous_input_location_string != input_location_string;

  if (has_new_string_input) {
    previous_input_location_string = input_location_string;
    return input_location_string;
  }

  return {};
}

bool LocationView::location_button(std::string location_name) {
  bool pressed = false;
  if (ImGui::Button(location_name.c_str())) {

    std::string selected_location_name = location_name;
    selected_location_name =
        selected_location_name.substr(0, selected_location_name.find(","));
    input_location_string.resize(MAX_INPUT_STRING_SIZE, '\0');
    input_location_string = selected_location_name;
    pressed = true;
  }

  return pressed;
}
