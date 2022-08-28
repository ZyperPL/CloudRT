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
#include "weather_entry.hpp"
#include "weather_entry_section_view.hpp"
#include "weather_entry_view.hpp"

#include "imgui_image_utils.hpp"

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

void configure_imgui(GLFWwindow *handle) {
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

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark();
  ImGuiStyle &style = ImGui::GetStyle();

  style.Alpha = 1.0;
  style.DisabledAlpha = 0.5;
  style.WindowPadding = ImVec2(8.0, 16.0);
  style.WindowRounding = 4.0;
  style.WindowBorderSize = 1.0;
  style.WindowMinSize = ImVec2(32.0, 32.0);
  style.WindowTitleAlign = ImVec2(0.5, 0.5);
  style.WindowMenuButtonPosition = ImGuiDir_Right;
  style.ChildRounding = 2.0;
  style.ChildBorderSize = 1.0;
  style.PopupRounding = 4.0;
  style.PopupBorderSize = 1.0;
  style.FramePadding = ImVec2(20.0, 6.0);
  style.FrameRounding = 2.0;
  style.FrameBorderSize = 0.0;
  style.ItemSpacing = ImVec2(4.0, 4.0);
  style.ItemInnerSpacing = ImVec2(4.0, 4.0);
  style.CellPadding = ImVec2(4.0, 2.0);
  style.IndentSpacing = 12.0;
  style.ColumnsMinSpacing = 6.0;
  style.ScrollbarSize = 14.0;
  style.ScrollbarRounding = 4.0;
  style.GrabMinSize = 10.0;
  style.GrabRounding = 4.0;
  style.TabRounding = 12.0;
  style.TabBorderSize = 1.0;
  style.TabMinWidthForCloseButton = 0.0;
  style.ColorButtonPosition = ImGuiDir_Right;
  style.ButtonTextAlign = ImVec2(0.5, 0.5);
  style.SelectableTextAlign = ImVec2(0.0, 0.5);

  style.Colors[ImGuiCol_Text] = ImVec4(1.0, 1.0, 1.0, 1.0);
  style.Colors[ImGuiCol_TextDisabled] =
      ImVec4(0.5921568870544434, 0.5921568870544434, 0.5921568870544434, 1.0);
  style.Colors[ImGuiCol_WindowBg] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_ChildBg] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_PopupBg] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_Border] =
      ImVec4(0.3058823645114899, 0.3058823645114899, 0.3058823645114899, 1.0);
  style.Colors[ImGuiCol_BorderShadow] =
      ImVec4(0.3058823645114899, 0.3058823645114899, 0.3058823645114899, 1.0);
  style.Colors[ImGuiCol_FrameBg] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_FrameBgHovered] =
      ImVec4(0.2841828167438507, 0.6303126811981201, 0.8712446689605713, 1.0);
  style.Colors[ImGuiCol_FrameBgActive] =
      ImVec4(0.2879220545291901, 0.6350106000900269, 0.8712446689605713, 1.0);
  style.Colors[ImGuiCol_TitleBg] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_TitleBgActive] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_TitleBgCollapsed] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_MenuBarBg] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_ScrollbarBg] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_ScrollbarGrab] =
      ImVec4(0.321568638086319, 0.321568638086319, 0.3333333432674408, 1.0);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] =
      ImVec4(0.3529411852359772, 0.3529411852359772, 0.3725490272045135, 1.0);
  style.Colors[ImGuiCol_ScrollbarGrabActive] =
      ImVec4(0.3529411852359772, 0.3529411852359772, 0.3725490272045135, 1.0);
  style.Colors[ImGuiCol_CheckMark] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_SliderGrab] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_SliderGrabActive] =
      ImVec4(0.6, 0.7666666686534882, 0.8843137383460999, 1.0);
  style.Colors[ImGuiCol_Button] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_ButtonHovered] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_ButtonActive] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_Header] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_HeaderHovered] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_HeaderActive] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_Separator] =
      ImVec4(0.3058823645114899, 0.3058823645114899, 0.3058823645114899, 1.0);
  style.Colors[ImGuiCol_SeparatorHovered] =
      ImVec4(0.3058823645114899, 0.3058823645114899, 0.3058823645114899, 1.0);
  style.Colors[ImGuiCol_SeparatorActive] =
      ImVec4(0.3058823645114899, 0.3058823645114899, 0.3058823645114899, 1.0);
  style.Colors[ImGuiCol_ResizeGrip] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_ResizeGripHovered] =
      ImVec4(0.2000000029802322, 0.2000000029802322, 0.2156862765550613, 1.0);
  style.Colors[ImGuiCol_ResizeGripActive] =
      ImVec4(0.321568638086319, 0.321568638086319, 0.3333333432674408, 1.0);
  style.Colors[ImGuiCol_Tab] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_TabHovered] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_TabActive] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_TabUnfocused] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_TabUnfocusedActive] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_PlotLines] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_PlotLinesHovered] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_PlotHistogram] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_PlotHistogramHovered] =
      ImVec4(0.1137254908680916, 0.5921568870544434, 0.9254902005195618, 1.0);
  style.Colors[ImGuiCol_TableHeaderBg] =
      ImVec4(0.1882352977991104, 0.1882352977991104, 0.2000000029802322, 1.0);
  style.Colors[ImGuiCol_TableBorderStrong] =
      ImVec4(0.3098039329051971, 0.3098039329051971, 0.3490196168422699, 1.0);
  style.Colors[ImGuiCol_TableBorderLight] =
      ImVec4(0.2274509817361832, 0.2274509817361832, 0.2470588237047195, 1.0);
  style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0, 0.0, 0.0, 0.0);
  style.Colors[ImGuiCol_TableRowBgAlt] =
      ImVec4(1.0, 1.0, 1.0, 0.05999999865889549);
  style.Colors[ImGuiCol_TextSelectedBg] =
      ImVec4(0.0, 0.4666666686534882, 0.7843137383460999, 1.0);
  style.Colors[ImGuiCol_DragDropTarget] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_NavHighlight] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);
  style.Colors[ImGuiCol_NavWindowingHighlight] =
      ImVec4(1.0, 1.0, 1.0, 0.699999988079071);
  style.Colors[ImGuiCol_NavWindowingDimBg] =
      ImVec4(0.800000011920929, 0.800000011920929, 0.800000011920929,
             0.2000000029802322);
  style.Colors[ImGuiCol_ModalWindowDimBg] =
      ImVec4(0.1450980454683304, 0.1450980454683304, 0.1490196138620377, 1.0);

  ImGui_ImplGlfw_InitForOpenGL(handle, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImVector<ImWchar> ranges;
  ImFontGlyphRangesBuilder builder;
  builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
  builder.AddText("zażółć gęślą jaźń");
  builder.AddText("◀▶◂▸");
  builder.BuildRanges(&ranges);
  io.Fonts->AddFontFromFileTTF("assets/iosevka-aile-medium.ttf", 16.0f, NULL,
                               ranges.Data);

  io.ConfigFlags &= ~ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplOpenGL3_NewFrame();
}

WindowMain::WindowMain() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return;

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

  configure_imgui(handle);

  render_texture =
      std::make_unique<Texture>(540 * 1.4, 540 * 1.4, Texture::Format::RGBA);
  clouds_texture = std::make_unique<Texture>(512, 512, Texture::Format::RGBA);

  clouds_texture_parameters.position = glm::vec3(0.0f);
  clouds_texture_parameters.width = clouds_texture->get_width();
  clouds_texture_parameters.height = clouds_texture->get_height();
  clouds_texture_parameters.time = 1.0f;
  clouds_texture_parameters.frequency = 2.0f;
  clouds_texture_parameters.octaves = 3.0f;
  clouds_texture_parameters.low_cut_l = 0.1f;
  clouds_texture_parameters.high_cut_l = 0.7f;
  clouds_texture_parameters.low_cut_m = 0.1f;
  clouds_texture_parameters.high_cut_m = 0.7f;
  clouds_texture_parameters.low_cut_h = 0.1f;
  clouds_texture_parameters.high_cut_h = 0.7f;

  render_parameters.width = render_texture->get_width();
  render_parameters.height = render_texture->get_height();
  render_parameters.camera_position = glm::vec3(0.0f, 1.0f, -1.0f);
  render_parameters.camera_rotation = glm::vec2(0.5f, 0.5f);
  render_parameters.light_direction = glm::vec3(0.6f, 0.65f, -0.8f);
  render_parameters.density = 0.9f;
}

void WindowMain::render() {

  glfwPollEvents();

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  static bool show_demo_window = false;
  static bool debug_enabled = true;
  static bool show_status_window = false;
  static bool show_help_window = false;
  static bool show_authors_window = false;

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
    if (ImGui::BeginMenu("File")) {
      ImGui::Separator();
      if (ImGui::MenuItem("Save as PNG")) {
        int ret = render_texture->save_to_file("image.png", FileFormat::PNG);
        printf("File save code: %d\n", ret);
      }
      if (ImGui::MenuItem("Save as HDR")) {
        int ret = render_texture->save_to_file("image.hdr", FileFormat::PNG);
        printf("File save code: %d\n", ret);
      }
      if (ImGui::MenuItem("Exit"))
        glfwSetWindowShouldClose(handle, GLFW_TRUE);
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Options")) {
      ImGui::Separator();
      if (ImGui::MenuItem("Enable debug", "", debug_enabled))
        debug_enabled = !debug_enabled;
      ImGui::Separator();
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Utilities")) {
      ImGui::Separator();
      if (ImGui::MenuItem("Show GUI demo window", "", show_demo_window))
        show_demo_window = !show_demo_window;
      if (ImGui::MenuItem("Show status window", "", show_status_window))
        show_status_window = !show_status_window;
      ImGui::Separator();
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("About")) {
      ImGui::Separator();
      if (ImGui::MenuItem("Help"))
        show_help_window = true;

      if (ImGui::MenuItem("Authors"))
        show_authors_window = true;
      ImGui::Separator();
      ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
  }

  ImGui::End();

  if (show_help_window) {
    ImGui::OpenPopup("Help");
    show_help_window = false;
  }

  if (show_authors_window) {
    ImGui::OpenPopup("Author");
    show_authors_window = false;
  }

  static ssize_t previous_index = -1;

  if (ImGui::BeginPopupModal("Help", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("1. Type location query in Location Window");
    ImGui::Text("2. Clouds are rendered in the Render Window");
    ImGui::Text("3. Change time and day in the Date & Time Window");
    ImGui::Text("4. Save an image via File menu");

    if (ImGui::Button("OK"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  if (ImGui::BeginPopupModal("Author", NULL,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Application created by:");
    ImGui::Text("Kacper Zybała (C) 2022");

    if (ImGui::Button("OK"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }

  if (show_demo_window)
    ImGui::ShowDemoWindow(&show_demo_window);

  if (ImGui::Begin("Location")) {
    location_controller.execute();
    if (location_controller.has_entry()) {
      date_time_controller.set_entry(location_controller.get_entry());

      static WeatherEntryView *view = new WeatherEntryView();
      view->render_ui(location_controller.get_entry());

      mapper.set_entry(location_controller.get_entry());
    }
  }
  ImGui::End();

  if (ImGui::Begin("Date & Time"))
    date_time_controller.execute();
  ImGui::End();

  if (ImGui::Begin("Weather")) {
    if (location_controller.has_entry()) {
      static WeatherEntrySectionView entry_section_view;

      const auto &entry = location_controller.get_entry();
      const auto index = date_time_controller.get_index();
      if (entry && index >= 0) {
        auto section_optional = entry->section(index);
        if (section_optional.has_value())
          entry_section_view.render_ui(section_optional.value());
        entry_section_view.render_range(entry, index, 12);
      }
    }
  }
  ImGui::End();

  if (previous_index == -1 ||
      previous_index != date_time_controller.get_index()) {

    // calculate params
    if (mapper.has_entry()) {
      mapper.calculate(static_cast<size_t>(date_time_controller.get_index()));
      if (mapper.get_render_parameters())
        render_parameters = *mapper.get_render_parameters();

      if (mapper.get_clouds_texture_parameters()) {
        clouds_texture_parameters = *mapper.get_clouds_texture_parameters();
        clouds_texture_parameters.width = clouds_texture->get_width();
        clouds_texture_parameters.height = clouds_texture->get_height();
        generate_cloud_noise(*clouds_texture, clouds_texture_parameters);
      }
    }
    previous_index = date_time_controller.get_index();
  }

  if (debug_enabled) {
    if (ImGui::Begin("Parameters")) {
      ImGui::Text("Render parameters");

      ImGui::DragFloat3("Camera position",
                        glm::value_ptr(render_parameters.camera_position), 1.0f,
                        -1000.0f, 1000.0f);

      ImGui::DragFloat2("Camera direction",
                        glm::value_ptr(render_parameters.camera_rotation),
                        0.01f, -1.0f, 1.0f);

      ImGui::DragFloat3("Sun direction",
                        glm::value_ptr(render_parameters.light_direction),
                        0.05f, -1.0f, 1.0f);

      ImGui::DragFloat3("Light color",
                        glm::value_ptr(render_parameters.light_color), 0.01f,
                        0.0f, 1.0f);

      ImGui::SliderFloat("Density", &render_parameters.density, 0.0f, 1.0f);
    }
    ImGui::End();

    if (ImGui::Begin("Clouds")) {
      ImGui::DragFloat("Frequency", &clouds_texture_parameters.frequency);
      ImGui::DragFloat("Octaves", &clouds_texture_parameters.octaves);
      ImGui::DragFloat("Low cut L", &clouds_texture_parameters.low_cut_l,
                       0.005f, 0.0f, 1.0f);
      ImGui::DragFloat("High cut L", &clouds_texture_parameters.high_cut_l,
                       0.005f, 0.0f, 1.0f);

      ImGui::DragFloat("Low cut M", &clouds_texture_parameters.low_cut_m,
                       0.005f, 0.0f, 1.0f);
      ImGui::DragFloat("High cut M", &clouds_texture_parameters.high_cut_m,
                       0.005f, 0.0f, 1.0f);

      ImGui::DragFloat("Low cut H", &clouds_texture_parameters.low_cut_h,
                       0.005f, 0.0f, 1.0f);
      ImGui::DragFloat("High cut H", &clouds_texture_parameters.high_cut_h,
                       0.005f, 0.0f, 1.0f);

      ImGui::DragFloat3("Position",
                        glm::value_ptr(clouds_texture_parameters.position),
                        0.001f, 0.0f, 100.0f);

      static bool auto_regenerate = false;
      if (ImGui::Button("Regenerate") || auto_regenerate) {
        clouds_texture_parameters.width = clouds_texture->get_width();
        clouds_texture_parameters.height = clouds_texture->get_height();
        generate_cloud_noise(*clouds_texture, clouds_texture_parameters);
      }
      ImGui::SameLine();
      ImGui::Checkbox("Continously", &auto_regenerate);

      if (clouds_texture) {
        ImGui::Image(*clouds_texture.get());
      }
    }
    ImGui::End();
  }

  if (ImGui::Begin("Render")) {
    if (render_texture && clouds_texture) {
      render_parameters.width = render_texture->get_width();
      render_parameters.height = render_texture->get_height();
      render_parameters.time += 0.1f;
      launch_render(*render_texture, *clouds_texture, render_parameters);
      ImGui::Text("size = %zu x %zu", render_texture->get_width(),
                  render_texture->get_height());

      ImGui::Image(*render_texture.get());
    } else {
      ImGui::Text("Render texture not defined!");
    }
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
