#include "weather_entry.hpp"

#include <type_traits>

WeatherEntry::WeatherEntry(nlohmann::json json, std::string name) : name{name} {
  if (!json.contains("data_1h"))
    return;

  const auto data = json["data_1h"];

  if (json.contains("metadata")) {
    const auto metadata = json["metadata"];
    this->latitude = metadata["latitude"].get<double>();
    this->longitude = metadata["longitude"].get<double>();
    this->height = metadata["height"].get<double>();
    this->modelrun_timestamp = metadata["modelrun_utc"].get<int>();
  } else {
    assert(!json.contains("metadata"));
  }

  const size_t count = data["time"].size();
  for (size_t i = 0; i < count; ++i) {
    const auto load_data = [&data, &i](const std::string key, auto &container,
                                       auto fallback) {
      const auto &value = data[key][i];
      try {
        container.push_back(value.get<decltype(fallback)>());
      } catch (...) {
        container.push_back(fallback);
      }
    };

    const long int DEFAULT_TIMESTAMP = 0;
    const int DEFAULT_INT = 0;
    const double DEFAULT_DOUBLE = 0.0;

    load_data("time", timestamps, DEFAULT_TIMESTAMP);
    load_data("winddirection_80m", winddirection, DEFAULT_INT);
    load_data("windspeed_80m", windspeed, DEFAULT_DOUBLE);
    load_data("surfaceairpressure", surfaceairpressure, DEFAULT_DOUBLE);
    load_data("lowclouds", lowclouds, DEFAULT_DOUBLE);
    load_data("midclouds", midclouds, DEFAULT_DOUBLE);
    load_data("highclouds", highclouds, DEFAULT_DOUBLE);
    load_data("cloudice", cloudice, DEFAULT_DOUBLE);
    load_data("cloudwater", cloudwater, DEFAULT_DOUBLE);
  }
}
