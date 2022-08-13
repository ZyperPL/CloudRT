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

  const auto t = data["time"][0].get<const long int>();

  auto time = std::gmtime(&t);
  printf("Loading entry starting at %02d:%02d:%02d %d.%02d.%04d\n",
         time->tm_hour, time->tm_min, time->tm_sec, time->tm_mday,
         time->tm_mon + 1, time->tm_year + 1900);

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

    load_data("time", timestamps, 0);
    load_data("winddirection_80m", winddirection, 0);
    load_data("windspeed_80m", windspeed, 0.0);
    load_data("surfaceairpressure", surfaceairpressure, 0.0);
    load_data("lowclouds", lowclouds, 0.0);
    load_data("midclouds", midclouds, 0.0);
    load_data("highclouds", highclouds, 0.0);
    load_data("cloudice", cloudice, 0.0);
    load_data("cloudwater", cloudwater, 0.0);
  }
}
