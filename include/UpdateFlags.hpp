#pragma once
#include <stdint.h>

enum UpdateFlags : uint8_t {
  None    = 0,
  Render  = 1 << 0,
  Zoom    = 1 << 1,
  Resize  = 1 << 2
};

inline UpdateFlags operator|(UpdateFlags a, UpdateFlags b) {
  return static_cast<UpdateFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline UpdateFlags operator&(UpdateFlags a, UpdateFlags b) {
  return static_cast<UpdateFlags>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

inline UpdateFlags& operator|=(UpdateFlags& a, UpdateFlags b) {
  a = a | b;
  return a;
}

inline UpdateFlags& operator&=(UpdateFlags& a, UpdateFlags b) {
  a = a & b;
  return a;
}

inline UpdateFlags operator~(UpdateFlags a) {
  return static_cast<UpdateFlags>(~static_cast<uint8_t>(a));
}
