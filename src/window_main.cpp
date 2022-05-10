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

#include "glm/gtc/type_ptr.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) &&                                 \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "cloud_noise.hpp"
#include "render.hpp"
#include "texture.hpp"

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void GLAPIENTRY gl_message_callback(GLenum, GLenum type, GLuint,
                                    GLenum severity, GLsizei,
                                    const GLchar *message, const void *) {
  if (type == GL_DEBUG_TYPE_OTHER_ARB)
    return;

  fprintf(stderr,
          "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
          (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity,
          message);
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

  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(gl_message_callback, 0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(handle, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  io.Fonts->AddFontFromFileTTF("assets/iosevka-aile-medium.ttf", 14.0f);

  ImGui_ImplOpenGL3_NewFrame();
  render_texture = std::make_unique<Texture>(640, 480);
  clouds_texture = std::make_unique<Texture>(2048, 2048);
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

  ImGui::Begin("Parameters");
  ImGui::Text("Render parameters");

  static float t = 0.0f;
  t += 0.01f;
  ImGui::Text("Time: %f\n", t);

  static glm::vec3 camera_position(20.0f, 18.0f, -50.0f);
  ImGui::DragFloat3("Camera position", glm::value_ptr(camera_position), 1.0f,
                    -100.0f, 100.0f);

  static glm::vec3 light_position(0.0f, 0.0f, 0.0f);
  ImGui::DragFloat3("Light position", glm::value_ptr(light_position), 1.0f,
                    -100.0f, 100.0f);

  static glm::vec3 light_color(0.0f, 0.0f, 0.0f);
  ImGui::DragFloat3("Light color", glm::value_ptr(light_color), 1000.0f, 900.0f,
                    850.0f);

  ImGui::End();

  ImGui::Begin("Clouds");
  if (clouds_texture) {
    static float frequency = 1000.0f;
    static float octaves = 1.0f;
    ImGui::DragFloat("Frequency", &frequency);
    ImGui::DragFloat("Octaves", &octaves);

    static glm::vec3 noise_pos(0.0f, 0.0f, 0.0f);
    ImGui::DragFloat3("Position", glm::value_ptr(noise_pos), 0.001f, 0.0f, 100.0f);

    CloudsRenderParameters clouds_parameters;
    clouds_parameters.position = noise_pos;
    clouds_parameters.width =  clouds_texture->get_width();
    clouds_parameters.height = clouds_texture->get_height();
    clouds_parameters.time = t;
    clouds_parameters.frequency = frequency;
    clouds_parameters.octaves = octaves;
    generate_cloud_noise(*clouds_texture, clouds_parameters);
    clouds_texture->update();
    ImVec2 avail_size = ImGui::GetContentRegionAvail();
    ImGui::Image(
        (void *)(intptr_t)clouds_texture->get_id(),
        avail_size);
  }
  ImGui::End();

  ImGui::Begin("Render");

  if (render_texture && clouds_texture) {
    RenderParameters parameters;
    parameters.width = render_texture->get_width();
    parameters.height = render_texture->get_height();
    parameters.time = t;
    parameters.camera_position = camera_position;
    parameters.light_position = light_position;
    parameters.light_color = light_color;
    launch_render(*render_texture, *clouds_texture, parameters);
    ImGui::Text("size = %zu x %zu", render_texture->get_width(),
                render_texture->get_height());

    render_texture->update();
    ImGui::Image(
        (void *)(intptr_t)render_texture->get_id(),
        ImVec2(render_texture->get_width(), render_texture->get_height()));
  } else {
    ImGui::Text("Render texture not defined!");
  }
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
