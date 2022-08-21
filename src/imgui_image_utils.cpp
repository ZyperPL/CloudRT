#include "imgui_image_utils.hpp"

namespace ImGui {
void Image(Texture &texture) {
  ImVec2 avail_size = ImGui::GetContentRegionAvail();
  if (avail_size.x < avail_size.y)
    avail_size =
        ImVec2(avail_size.x, avail_size.x * (texture.get_aspect_ratio()));
  else
    avail_size = ImVec2(avail_size.y * (1.0f / texture.get_aspect_ratio()),
                        avail_size.y);
  ImGui::Image((void *)(intptr_t)texture.get_id(), avail_size);
}
} // namespace ImGui
