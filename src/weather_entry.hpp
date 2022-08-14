#pragma once

#include <chrono>

#include <nlohmann/json.hpp>

class WeatherEntry {
public:
  WeatherEntry(nlohmann::json json, std::string name);

  const std::string &get_name() const { return name; }
  const double &get_latitude() const { return latitude; }
  const double &get_longitude() const { return longitude; }
  const double &get_height() const { return height; }

  const int &get_modelrun_timestamp() const { return modelrun_timestamp; }

  const std::vector<long int> &get_timestamps() const { return timestamps; }
  const std::vector<int> &get_winddirection() const { return winddirection; }

  const std::vector<double> &get_windspeed() const { return windspeed; }
  const std::vector<double> &get_surfaceairpressure() const {
    return surfaceairpressure;
  }
  const std::vector<double> &get_lowclouds() const { return lowclouds; }
  const std::vector<double> &get_midclouds() const { return midclouds; }
  const std::vector<double> &get_highclouds() const { return highclouds; }
  const std::vector<double> &get_cloudwater() const { return cloudwater; }
  const std::vector<double> &get_cloudice() const { return cloudice; }

  size_t count() const { return timestamps.size(); }

private:
  std::string name;
  double latitude;
  double longitude;
  double height;
  int modelrun_timestamp;
  std::vector<long int> timestamps;
  std::vector<int> winddirection;
  std::vector<double> windspeed;
  std::vector<double> surfaceairpressure;
  std::vector<double> lowclouds;
  std::vector<double> midclouds;
  std::vector<double> highclouds;
  std::vector<double> cloudwater;
  std::vector<double> cloudice;
};
