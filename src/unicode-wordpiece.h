/** Normalizes uncased WordPiece input without discarding spacing or enclosing marks. */
#pragma once
#include <cstdint>
#include <vector>

std::vector<uint32_t> unicode_wordpiece_normalize(const std::vector<uint32_t> & input);
