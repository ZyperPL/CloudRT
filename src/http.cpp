#include "http.hpp"

#include <nlohmann/json.hpp>

#define PAR_EASYCURL_IMPLEMENTATION
#include "par/par_easycurl.h"

#include <curl/curl.h>

namespace HTTP {
char *url_escape_string(const char *str, size_t len) {
  CURL *curl = curl_easy_init();
  if (curl) {
    char *output = curl_easy_escape(curl, str, len);
    curl_easy_cleanup(curl);
    if (output) {
      return output;
    }
  }
  return nullptr;
}

std::string url_escape_string(std::string str) {
  char *cstr = url_escape_string(str.c_str(), str.size());
  std::string output{cstr};
  free(cstr);
  return output;
}

void QueryParameters::add(std::string key, std::string value) {
  values[key] = value;
}

void QueryParameters::remove(std::string key) { values.erase(key); }

std::string QueryParameters::string() {
  std::string parameters_string{};
  for (const auto &[key, value] : values) {
    parameters_string += HTTP::url_escape_string(key);
    parameters_string += "=";
    parameters_string += HTTP::url_escape_string(value);
  }

  return parameters_string;
}

std::optional<std::shared_ptr<DataObject>> get(const char *url) {
  par_easycurl_init(0);

  std::shared_ptr<DataObject> data_object = std::make_shared<DataObject>();
  int r = par_easycurl_to_memory(url, &data_object->data, &data_object->size);
  if (r == 1) {
    data_object->data[data_object->size] = '\0';
    return data_object;
  }

  return {};
}

std::optional<std::shared_ptr<DataObject>> get(const char *url,
                                               QueryParameters &parameters) {
  std::string url_with_parameters{url};
  url_with_parameters += "?";
  url_with_parameters += parameters.string();
  return HTTP::get(url_with_parameters.c_str());
}

} // namespace HTTP
