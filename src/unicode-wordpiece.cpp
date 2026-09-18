/** Applies canonical decomposition and ordering before WordPiece's Mn-only accent removal.
 * Generic tokenizer normalization is intentionally separate: BERT needs every
 * decomposed codepoint, including retained Mc and Me marks, rather than a base-letter projection.
 */
#include "unicode-wordpiece.h"
#include "unicode-wordpiece-data.h"
#include <algorithm>
#include <iterator>

namespace {
using namespace wordpiece_unicode_data;

uint8_t combining(uint32_t cp) {
    if (cp < 0x300) return 0;
    const auto it = std::lower_bound(std::begin(combining_classes), std::end(combining_classes), cp,
        [](const combining_class & item, uint32_t value) { return item.codepoint < value; });
    return it != std::end(combining_classes) && it->codepoint == cp ? it->value : 0;
}

bool nonspacing(uint32_t cp) {
    if (cp < 0x300) return false;
    const auto it = std::lower_bound(std::begin(nonspacing_marks), std::end(nonspacing_marks), cp,
        [](const mark_range & item, uint32_t value) { return item.last < value; });
    return it != std::end(nonspacing_marks) && it->first <= cp;
}
}

std::vector<uint32_t> unicode_wordpiece_nfd_strip_accents(const std::vector<uint32_t> & input) {
    using namespace wordpiece_unicode_data;
    std::vector<uint32_t> result;
    result.reserve(input.size());
    for (const uint32_t cp : input) {
        if (cp < 0x80) {
            result.push_back(cp);
            continue;
        }
        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            const uint32_t syllable = cp - 0xAC00;
            result.push_back(0x1100 + syllable / 588);
            result.push_back(0x1161 + (syllable % 588) / 28);
            if (syllable % 28) result.push_back(0x11A7 + syllable % 28);
            continue;
        }
        const auto it = std::lower_bound(std::begin(decompositions), std::end(decompositions), cp,
            [](const decomposition & item, uint32_t value) { return item.codepoint < value; });
        if (it != std::end(decompositions) && it->codepoint == cp) {
            result.insert(result.end(), it->values, it->values + it->count);
        } else {
            result.push_back(cp);
        }
    }

    // Sort non-starter runs before deleting Mn: a removed mark with class zero
    // remains an ordering boundary during NFD. Stable sorting avoids quadratic
    // insertion behavior for adversarial runs of combining characters.
    size_t start = 0;
    for (size_t i = 0; i <= result.size(); ++i) {
        if (i == result.size() || combining(result[i]) == 0) {
            std::stable_sort(result.begin() + start, result.begin() + i,
                [](uint32_t a, uint32_t b) { return combining(a) < combining(b); });
            start = i + 1;
        }
    }
    result.erase(std::remove_if(result.begin(), result.end(), nonspacing), result.end());
    return result;
}
