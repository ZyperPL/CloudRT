#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>

//#include "par/par_easycurl.h"

namespace HTTP {
struct DataObject {
  unsigned char *data{nullptr};

  int size{0};

  ~DataObject() { free(data); }
};

struct QueryParameters {
  std::map<std::string, std::string> values;

  void add(std::string key, std::string value);
  void remove(std::string key);

  std::string string();
};

std::optional<std::shared_ptr<DataObject>> get(const char *url);
std::optional<std::shared_ptr<DataObject>> get(const char *url,
                                               QueryParameters &parameters);

char *url_escape_string(const char *str, size_t len);
std::string url_escape_string(std::string str);
} // namespace HTTP
