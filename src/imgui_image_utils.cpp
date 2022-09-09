#include "imgui_image_utils.hpp"

namespace ImGui {
void Image(Texture &texture) {
  ImVec2 asize = ImGui::GetContentRegionAvail();
  asize.x -= 5.0f;
  asize.y -= 6.0f;
  ImVec2 nsize = asize;

  const float aspect = texture.get_aspect_ratio();
  const float inv_aspect = 1.0f / texture.get_aspect_ratio();

  nsize.y = asize.x * inv_aspect;
  if (nsize.y > asize.y) {
    nsize.y = asize.y;
    nsize.x = asize.y * aspect;
  }

  ImGui::Image((void *)(intptr_t)texture.get_id(), nsize);
}
} // namespace ImGui
