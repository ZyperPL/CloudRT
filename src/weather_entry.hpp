#pragma once

#include <chrono>

#include <nlohmann/json.hpp>

class WeatherEntry
{
public:
  WeatherEntry(nlohmann::json json, std::string name);

private:
  std::string name;
  double latitude;
  double longitude;
  double height;
  int modelrun_timestamp;
  std::vector<int> timestamps;
  std::vector<int> winddirection;
  std::vector<int> windspeed;
  std::vector<double> surfaceairpressure;
  std::vector<double> lowclouds;
  std::vector<double> midclouds;
  std::vector<double> highclouds;
  std::vector<double> cloudwater;
  std::vector<double> cloudice;
};
