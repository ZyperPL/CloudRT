#pragma once

#include <memory>

struct GLFWwindow;
class Texture;

class WindowMain {
public:
  WindowMain();
  ~WindowMain();
  void render();
  bool is_open();

private:
  GLFWwindow *handle{nullptr};
  std::unique_ptr<Texture> render_texture;
  std::unique_ptr<Texture> clouds_texture;
};
