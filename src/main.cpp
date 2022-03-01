#include <cstdio>

#include "glm/glm.hpp"
#include "test.hpp"

#include "window_main.hpp"
#include "texture.hpp"

int main() {
  printf("Ready.\n");
  start_cuda_hello();

  glm::vec3 vec{1.0, 2.0, 3.0};
  printf("Length: %24.12f\n", glm::length(vec));

  WindowMain window;

  while (window.is_open())
    window.render();

  return 0;
}
