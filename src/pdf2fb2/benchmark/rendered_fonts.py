"""Fonts for rendered training pages, chosen by the script of the text (any language of that script works).

Fonts are looked up in ``models/fonts`` (downloaded by ``scripts/download_verifier_assets.sh``, all SIL Open Font
License) and, as a fallback, among common system fonts, so rendering works before the download too.
"""

import unicodedata
from collections import Counter
from pathlib import Path

from pdf2fb2.project_paths import MODELS_DIRECTORY

FONTS_DIRECTORY = MODELS_DIRECTORY / "fonts"
SCRIPT_FONTS = {
    "latin": ("NotoSerif-Regular.ttf", "EBGaramond-Regular.ttf", "IMFellEnglish-Regular.ttf",
              "OldStandardTT-Regular.ttf", "PTSerif-Regular.ttf"),
    "cyrillic": ("NotoSerif-Regular.ttf", "OldStandardTT-Regular.ttf", "PTSerif-Regular.ttf",
                 "EBGaramond-Regular.ttf"),
    "greek": ("NotoSerif-Regular.ttf", "EBGaramond-Regular.ttf"),
    "cjk": ("NotoSerifCJKjp-Regular.otf",),
    "fraktur": ("UnifrakturMaguntia-Book.ttf",),
}
SYSTEM_FONTS = {
    "latin": ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
              "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
              "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"),
    "cyrillic": ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                 "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf"),
    "greek": ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",),
    "cjk": ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",),
    "fraktur": (),
}
SCRIPT_PREFIXES = (("CJK", "cjk"), ("HIRAGANA", "cjk"), ("KATAKANA", "cjk"), ("CYRILLIC", "cyrillic"),
                   ("GREEK", "greek"), ("LATIN", "latin"))


def text_script(text_value: str) -> str:
    """Dominant script of a text.

    Args:
        text_value: Text to inspect.

    Returns:
        ``latin``, ``cyrillic``, ``greek`` or ``cjk`` (``latin`` when no letter is recognised).
    """
    script_counts: Counter = Counter()
    for text_char in text_value:
        if text_char.isalpha():
            char_name = unicodedata.name(text_char, "")
            script_counts.update(script_name for name_prefix, script_name in SCRIPT_PREFIXES
                                 if char_name.startswith(name_prefix))
    return script_counts.most_common(1)[0][0] if script_counts else "latin"


def available_fonts(script_name: str, fonts_directory: Path = FONTS_DIRECTORY) -> list[Path]:
    """Font files that can render a script.

    Usage:
        Called by ``rendered_pages`` for every page.

    Args:
        script_name: Script from ``text_script`` (or ``fraktur``).
        fonts_directory: Directory of the downloaded fonts.

    Returns:
        Existing font files (downloaded ones first; may be empty).
    """
    downloaded = [fonts_directory / font_name for font_name in SCRIPT_FONTS.get(script_name, ())]
    system_fonts = [Path(font_path) for font_path in SYSTEM_FONTS.get(script_name, ())]
    return [font_path for font_path in downloaded + system_fonts if font_path.is_file()]
