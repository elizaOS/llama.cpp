/** Normalizes uncased WordPiece input without discarding spacing or enclosing marks. */
#pragma once
#include <cstdint>
#include <vector>

std::vector<uint32_t> unicode_wordpiece_nfd_strip_accents(const std::vector<uint32_t> & input);
