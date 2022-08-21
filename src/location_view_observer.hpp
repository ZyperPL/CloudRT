#pragma once

#include <string>

class LocationViewObserver
{
public:
  virtual void onLocationSelected(size_t) = 0;
  virtual void onLocationInputTextChanged(const std::string&) = 0;
  virtual void onButtonPressed() = 0;
};
