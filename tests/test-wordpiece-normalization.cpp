/** Exercises uncased WordPiece Unicode normalization and real BGE vocabulary admission. */
#include "llama.h"
#include "unicode-wordpiece.h"
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int size = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, false);
    check(size > 0, "BGE must produce special tokens");
    std::vector<llama_token> tokens(size);
    check(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, true, false) == size,
          "Token count changed between admission and allocation");
    return tokens;
}

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "Usage: %s BGE-vocab-or-model.gguf\n", argv[0]); return 2; }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 1;
    int status = 0;
    try {
        const auto normalize = [](const std::u32string & text) {
            return unicode_wordpiece_normalize(std::vector<uint32_t>(text.begin(), text.end()));
        };
        check(normalize(U"Café") == normalize(U"Cafe"), "Precomposed acute not stripped");
        check(normalize(U"Cafe\u0301") == normalize(U"Cafe"), "Decomposed Mn not stripped");
        check(normalize(U"Café\u0301\u0301") == normalize(U"Cafe"), "Repeated Mn not stripped");
        check(normalize(U"cafe\u0903") == std::vector<uint32_t>({'c','a','f','e',0x0903}), "Mc removed");
        check(normalize(U"cafe\u20dd") == std::vector<uint32_t>({'c','a','f','e',0x20dd}), "Me removed");
        check(normalize(U"\u09cb") == std::vector<uint32_t>({0x09c7,0x09be}), "NFD lost a spacing mark");
        check(normalize(U"\uac01") == std::vector<uint32_t>({0x1100,0x1161,0x11a8}), "Hangul NFD incomplete");
        check(normalize(U"x\u302e\u0301\u1715") == std::vector<uint32_t>({'x',0x1715,0x302e}), "Retained Mc ordering incorrect");

        for (char32_t control : {U'\0', U'\1', U'\uFFFD'}) {
            const std::u32string text{U'x', U'\u302E', control, U'\u1715'};
            check(normalize(text) == std::vector<uint32_t>({'x',0x1715,0x302e}),
                  "Removed control changed canonical ordering");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto cafe = tokenize(vocab, "cafe");
        check(cafe == std::vector<llama_token>({101,7668,102}), "Unexpected BGE vocabulary");
        const std::vector<std::string> variants = {u8"Café", u8"Cafe\u0301", u8"Café\u0301\u0301"};
        for (const auto & text : variants) check(tokenize(vocab, text) == cafe, "BGE accent token mismatch");
        check(tokenize(vocab, u8"cafe\u0903") != cafe, "BGE swallowed spacing mark");
        check(tokenize(vocab, u8"cafe\u20dd") != cafe, "BGE swallowed enclosing mark");
        check(tokenize(vocab, std::string("before\0after",12)) == tokenize(vocab,"beforeafter"), "NUL input lost its tail");
        std::string boundary;
        for (int i=0; i<510; ++i) boundary += "word ";
        check(tokenize(vocab,boundary).size() == 512, "512-token admission changed");
        check(tokenize(vocab,boundary+"word ").size() == 513, "Tokenizer truncated oversized input");
        std::atomic<bool> concurrent_ok{true};
        std::vector<std::thread> threads;
        for (int i=0; i<8; ++i) threads.emplace_back([&] {
            for (int n=0; n<20; ++n) {
                for (const auto & text : variants) {
                    if (tokenize(vocab,text) != cafe) concurrent_ok = false;
                }
            }
        });
        for (auto & thread : threads) thread.join();
        check(concurrent_ok, "Concurrent WordPiece normalization diverged");
        std::puts("WordPiece Unicode and BGE admission passed (8 concurrent readers)");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); status = 1;
    }
    llama_model_free(model);
    return status;
}
