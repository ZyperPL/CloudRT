#include "window_main.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/glew.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) &&                                 \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "texture.hpp"
#include "test.hpp"

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

WindowMain::WindowMain() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return;

#if defined(IMGUI_IMPL_OPENGL_ES2)
  const char *glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

  handle = glfwCreateWindow(1920, 1080, "CloudRT", NULL, NULL);
  if (!handle)
    return;
  glfwMakeContextCurrent(handle);

  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
  }
  fprintf(stdout, "GLEW version: %s\n", glewGetString(GLEW_VERSION));

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(handle, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  io.Fonts->AddFontFromFileTTF("assets/iosevka-aile-medium.ttf", 14.0f);
}

void WindowMain::render() {

  glfwPollEvents();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  static bool show_demo_window = false;

  const ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("DockSpace", NULL, window_flags);

  ImGui::PopStyleVar(3);
  ImGuiIO &io = ImGui::GetIO();
  if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
  }
  if (ImGui::BeginMenuBar()) {
    if (ImGui::BeginMenu("Options")) {
      ImGui::Separator();
      ImGui::MenuItem("Test");
      if (ImGui::MenuItem("Show Demo Window", "", show_demo_window))
        show_demo_window = !show_demo_window;
      ImGui::Separator();
      ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
  }
  ImGui::End();

  if (show_demo_window)
    ImGui::ShowDemoWindow(&show_demo_window);

  ImGui::Begin("Window");
  ImGui::Text("Test");
  ImGui::End();

  const size_t w = 1024;
  const size_t h = 1024;
  static Texture texture(w, h);
  auto pbo = texture.get_pbo_resource();
  launch_render(pbo, w, h);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

  ImGui::Begin("OpenGL Texture Text");
  ImGui::Text("size = %zu x %zu", texture.get_width(), texture.get_height());
  ImGui::Image((void *)(intptr_t)texture.get_id(),
               ImVec2(texture.get_width(), texture.get_height()));
  ImGui::End();

  ImGui::Render();
  int framebuffer_width, framebuffer_height;
  glfwGetFramebufferSize(handle, &framebuffer_width, &framebuffer_height);
  glViewport(0, 0, framebuffer_width, framebuffer_height);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    GLFWwindow *backup_current_context = glfwGetCurrentContext();
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }

  glfwSwapBuffers(handle);
}

bool WindowMain::is_open() { return handle && !glfwWindowShouldClose(handle); }

WindowMain::~WindowMain() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (handle)
    glfwDestroyWindow(handle);
  glfwTerminate();
}
