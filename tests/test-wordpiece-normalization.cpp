/** Exercises uncased WordPiece Unicode normalization and real BGE vocabulary admission. */
#include "llama.h"
#include "unicode-wordpiece.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special = false) {
    const int size = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, parse_special);
    check(size > 0, "BGE must produce special tokens");
    std::vector<llama_token> tokens(size);
    check(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, true, parse_special) == size,
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

        check(normalize(U"ΟΣ") == normalize(U"ος"), "Final Sigma not contextual");
        check(normalize(U"ΟΣΑ") == normalize(U"οσα"), "Medial Sigma became final");
        check(normalize(U"Σ") == normalize(U"σ"), "Isolated Sigma became final");
        check(normalize(U"AΣ'A") == normalize(U"aσ'a"), "Case-ignorable punctuation ended casing context");
        check(normalize(U"AΣ\u0301A") == normalize(U"aσa"), "Case-ignorable Mn ended casing context");
        check(normalize(U"A\u0301Σ") == normalize(U"aς"), "Preceding Mn lost casing context");

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const auto cafe = tokenize(vocab, "cafe");
        check(cafe == std::vector<llama_token>({101,7668,102}), "Unexpected BGE vocabulary");
        const std::vector<std::string> variants = {u8"Café", u8"Cafe\u0301", u8"Café\u0301\u0301"};
        for (const auto & text : variants) check(tokenize(vocab, text) == cafe, "BGE accent token mismatch");
        check(tokenize(vocab, u8"cafe\u0903") != cafe, "BGE swallowed spacing mark");
        check(tokenize(vocab, u8"cafe\u20dd") != cafe, "BGE swallowed enclosing mark");
        check(tokenize(vocab, std::string("before\0after",12)) == tokenize(vocab,"beforeafter"), "NUL input lost its tail");
        check(tokenize(vocab, u8"ΟΣ") == std::vector<llama_token>({101,1169,19579,102}), "BGE Final Sigma IDs differ");
        check(tokenize(vocab, u8"ΟΔΟΣ") == std::vector<llama_token>({101,1169,29722,15297,102}), "BGE contextual word IDs differ");
        const std::vector<llama_token> unknown = {101,100,102};
        check(tokenize(vocab, std::string(100, 'z')) != unknown, "100-scalar word was rejected");
        check(tokenize(vocab, std::string(101, 'z')) == unknown, "101-scalar word was partially encoded");
        check(tokenize(vocab, std::string(200, 'z')) == unknown, "Oversized word did not produce one UNK");
        std::string greek_word;
        for (int i=0; i<100; ++i) greek_word += u8"α";
        check(tokenize(vocab, greek_word) != unknown, "WordPiece counted UTF-8 bytes instead of scalars");
        check(tokenize(vocab, greek_word + u8"α") == unknown, "Unicode scalar limit was ignored");
        const auto special = tokenize(vocab, "[CLS] [MASK] [SEP]", true);
        check(special == std::vector<llama_token>({101,101,103,102,102}), "Added-token parsing diverged");
        check(tokenize(vocab, "[CLS] [MASK] [SEP]") != special, "Default tokenizer special parsing changed");

        // Exercise the loader's real GGUF metadata override path as well as
        // the metadata-absent standard default used by the pinned fixture.
        llama_model_kv_override overrides[2]{};
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strcpy(overrides[0].key, "tokenizer.ggml.max_input_chars_per_word");
        overrides[0].val_i64 = 4;
        params.kv_overrides = overrides;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> custom(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        check(custom != nullptr, "Custom WordPiece metadata failed to load");
        check(tokenize(llama_model_get_vocab(custom.get()), "hello") == unknown, "Custom word limit ignored");
        check(tokenize(llama_model_get_vocab(custom.get()), "cafe") == cafe, "Custom limit rejected boundary word");
        overrides[0].val_i64 = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> zero(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        check(zero != nullptr, "Zero WordPiece limit failed to load");
        check(tokenize(llama_model_get_vocab(zero.get()), "cafe") == unknown, "Zero word limit ignored");

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
