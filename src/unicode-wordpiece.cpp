/** Cleans and canonically decomposes BERT input before Mn removal and scalar lowercase.
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

template<size_t N>
bool in_ranges(uint32_t cp, const codepoint_range (& ranges)[N]) {
    const auto it = std::lower_bound(std::begin(ranges), std::end(ranges), cp,
        [](const codepoint_range & item, uint32_t value) { return item.last < value; });
    return it != std::end(ranges) && it->first <= cp;
}

bool nonspacing(uint32_t cp) {
    return cp >= 0x300 && in_ranges(cp, nonspacing_marks);
}
}

std::vector<uint32_t> unicode_wordpiece_normalize(const std::vector<uint32_t> & input) {
    using namespace wordpiece_unicode_data;
    std::vector<uint32_t> cleaned;
    cleaned.reserve(input.size());
    for (uint32_t cp : input) {
        // BERT clean_text precedes NFD. Deleted controls cannot act as
        // canonical-ordering boundaries.
        if (cp < 0x80) {
            if (cp == 9 || cp == 10 || cp == 13) cleaned.push_back(0x20);
            else if (cp >= 0x20 && cp < 0x7F) cleaned.push_back(cp);
            continue;
        }
        if (cp == 0xFFFD || in_ranges(cp, controls)) continue;
        cleaned.push_back(in_ranges(cp, spaces) ? 0x20 : cp);
    }

    std::vector<uint32_t> result;
    result.reserve(cleaned.size());
    for (const uint32_t cp : cleaned) {
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
    // Hugging Face NormalizedString::lowercase maps each scalar separately.
    // Whole-string Final_Sigma casing changes BGE token IDs and vector space.
    std::vector<uint32_t> lowered;
    lowered.reserve(result.size());
    for (const uint32_t cp : result) {
        const auto it = std::lower_bound(std::begin(lowercase_mappings), std::end(lowercase_mappings), cp,
            [](const decomposition & item, uint32_t value) { return item.codepoint < value; });
        if (it != std::end(lowercase_mappings) && it->codepoint == cp)
            lowered.insert(lowered.end(), it->values, it->values + it->count);
        else lowered.push_back(cp);
    }
    return lowered;
}
