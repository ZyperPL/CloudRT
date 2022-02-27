#pragma once

struct GLFWwindow;

class WindowMain {
public:
  WindowMain();
  ~WindowMain();
  void render();
  bool is_open();

private:
  GLFWwindow *handle{nullptr};
};
