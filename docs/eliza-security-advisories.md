# Eliza fork: upstream security advisories

This fork is based on ggml-org llama.cpp `b9159`. Upstream security fixes
reach it only by a rebase or a cherry-pick. Before you move the elizaOS
`plugins/plugin-local-inference/native/llama.cpp` gitlink to a new fork
commit, do these steps:

1. Read the ggml-org advisories:
   `gh api repos/ggml-org/llama.cpp/security-advisories`.
2. Search OSV for llama.cpp CVEs that are not in that list (many are filed
   only as CVEs): `https://api.osv.dev/v1/query` with the package
   `github.com/ggml-org/llama.cpp`.
3. For each fix commit, check `git merge-base --is-ancestor <fix> HEAD`.
   If it is missing, cherry-pick it with `git cherry-pick -x`.
4. Update the table below.

Keep `GGML_RPC` off in every Eliza build. The RPC server has an open
unauthenticated remote code execution advisory with no upstream fix.

| Advisory | Component | Status in this fork |
| --- | --- | --- |
| CVE-2026-107183 (use-after-free, chat PEG parser) | `common/chat-peg-parser.cpp` | Fixed: cherry-pick of ggml-org `dbe4c3e` (#29942). |
| CVE-2026-52132 (`/rerank` negative `top_n`, DoS) | `tools/server` | Fixed in the fork: a negative `top_n` returns HTTP 400. No ggml-org fix yet. |
| GHSA-j8rj-fmpv-wcxw / CVE-2026-34159 (RPC `GRAPH_COMPUTE`, RCE) | `ggml-rpc` | No upstream fix. Not built: `GGML_RPC` is off. |
| CVE-2026-86317 (RPC `deserialize_tensor` assertion) | `ggml-rpc` | No upstream fix. Not built: `GGML_RPC` is off. |
| GGUF parser and tokenizer advisories up to GHSA-96jg-mvhq-q7q7 | `ggml/src/gguf.cpp`, `src/llama-vocab.cpp` | Fixed in the `b9159` base. |
