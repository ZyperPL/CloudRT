#include <cstdio>

#include "glm/glm.hpp"

#include "window_main.hpp"
#include "texture.hpp"

int main() {
  printf("Ready.\n");

  WindowMain window;

  while (window.is_open())
    window.render();

  return 0;
}
