#!/usr/bin/env bash
# download_verifier_assets.sh - download the extra models and fonts used by the token-verifier improvements.
#
# Usage:
#   scripts/download_verifier_assets.sh [all|fonts|mdeberta|byt5]...
#   e.g. scripts/download_verifier_assets.sh fonts mdeberta
#
# Arguments:
#   fonts     SIL OFL fonts for rendered training pages -> models/fonts/
#             (Noto Serif, EB Garamond, IM Fell English, Old Standard TT, PT Serif, UnifrakturMaguntia,
#              Noto Serif CJK JP)
#   mdeberta  microsoft/mdeberta-v3-base -> models/mdeberta-v3-base (alternative sub-word backbone)
#   byt5      google/byt5-small          -> models/byt5-small       (byte-level backbone)
#             Both repos publish pickled weights only on main; the safetensors variants of their
#             conversion pull requests (refs/pr/5 and refs/pr/3) are downloaded instead.
#   all       everything above (default when no argument is given)
#
# Already used and not downloaded here: models/xlm-roberta-base (backbone and plausibility scorer),
# models/dinov2-small (word-crop embeddings), models/qwen3.5-9B (vision-language judge).
#
# Outputs:
#   Files under models/. Existing files are kept (re-run to complete an interrupted download).
#   Exits non-zero on the first failing download.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="${project_root}/models"
hf_command="${project_root}/workers/verifier_worker/.venv/bin/hf"

download_font() {
    # download_font <url> <file name>: fetch one font into models/fonts unless it already exists.
    local font_url="$1"
    local font_name="$2"
    local font_path="${models_root}/fonts/${font_name}"
    if [[ -s "${font_path}" ]]; then
        echo "font present: ${font_name}"
        return 0
    fi
    curl --fail --location --silent --show-error --proto '=https' --output "${font_path}.part" "${font_url}"
    mv "${font_path}.part" "${font_path}"
    echo "font downloaded: ${font_name}"
}

download_fonts() {
    # download_fonts: fetch every rendering font (SIL Open Font License).
    mkdir -p "${models_root}/fonts"
    download_font "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSerif/hinted/ttf/NotoSerif-Regular.ttf" \
        "NotoSerif-Regular.ttf"
    download_font "https://github.com/octaviopardo/EBGaramond12/raw/master/fonts/ttf/EBGaramond-Regular.ttf" \
        "EBGaramond-Regular.ttf"
    download_font "https://github.com/google/fonts/raw/main/ofl/imfellenglish/IMFeENrm28P.ttf" \
        "IMFellEnglish-Regular.ttf"
    download_font "https://github.com/google/fonts/raw/main/ofl/oldstandardtt/OldStandard-Regular.ttf" \
        "OldStandardTT-Regular.ttf"
    download_font "https://github.com/google/fonts/raw/main/ofl/ptserif/PT_Serif-Web-Regular.ttf" \
        "PTSerif-Regular.ttf"
    download_font "https://github.com/google/fonts/raw/main/ofl/unifrakturmaguntia/UnifrakturMaguntia-Book.ttf" \
        "UnifrakturMaguntia-Book.ttf"
    download_font "https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF/Japanese/NotoSerifCJKjp-Regular.otf" \
        "NotoSerifCJKjp-Regular.otf"
}

download_model() {
    # download_model <repo id> <revision> <directory name>: fetch safetensors weights, config and tokenizer files.
    local repo_id="$1"
    local revision_name="$2"
    local target_name="$3"
    if [[ ! -x "${hf_command}" ]]; then
        echo "missing ${hf_command}; run scripts/sync_environments.sh first" >&2
        return 1
    fi
    "${hf_command}" download "${repo_id}" --revision "${revision_name}" \
        --local-dir "${models_root}/${target_name}" \
        --include "*.json" --include "*.safetensors" --include "*.model"
    echo "model downloaded: ${repo_id} -> models/${target_name}"
}

main() {
    # main [targets...]: download the requested assets (all when none is given).
    local requested_targets=("$@")
    if [[ ${#requested_targets[@]} -eq 0 ]]; then
        requested_targets=(all)
    fi
    local target_name
    for target_name in "${requested_targets[@]}"; do
        case "${target_name}" in
            fonts) download_fonts ;;
            mdeberta) download_model "microsoft/mdeberta-v3-base" "refs/pr/5" "mdeberta-v3-base" ;;
            byt5) download_model "google/byt5-small" "refs/pr/3" "byt5-small" ;;
            all)
                download_fonts
                download_model "microsoft/mdeberta-v3-base" "refs/pr/5" "mdeberta-v3-base"
                download_model "google/byt5-small" "refs/pr/3" "byt5-small"
                ;;
            *)
                echo "unknown target: ${target_name} (expected all, fonts, mdeberta or byt5)" >&2
                return 2
                ;;
        esac
    done
}

main "$@"
