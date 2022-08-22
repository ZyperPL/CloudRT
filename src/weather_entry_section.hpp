#pragma once

#include <string>

struct WeatherEntrySection {
  std::string name;
  double latitude;
  double longitude;
  double height;
  long int timestamp;
  int wind_direction;
  double wind_speed;
  double surface_air_pressure;
  double low_clouds;
  double mid_clouds;
  double high_clouds;
  double cloud_water;
  double cloud_ice;
};
