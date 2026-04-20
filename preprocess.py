"""
STEP 2 — Text Normalization
Preprocessing module for Akkadian transliterations and English translations.
"""
import re
import unicodedata


# ── Subscript digit mapping ────────────────────────────────────────────────
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

# ── Hamza / Ayin / glottal stop characters to delete ──────────────────────
_GLOTTAL_CHARS = "\u02BE\u02BF\u02BC\u02BB\u0027\u2018\u2019"  # ʾ ʿ ʼ ʻ ' ' '

# ── Determinative patterns ─────────────────────────────────────────────────
# Common Sumerian determinatives in Akkadian texts
_DETERMINATIVE_PATTERNS = [
    # Before divine names
    (re.compile(r'\{d\}', re.IGNORECASE), '{d}'),
    (re.compile(r'\bDINGIR\b'), '{d}'),
    # Geographic
    (re.compile(r'\{ki\}', re.IGNORECASE), '{ki}'),
    # Personal name (masculine)
    (re.compile(r'\{m\}', re.IGNORECASE), '{m}'),
    # Personal name (feminine)
    (re.compile(r'\{f\}', re.IGNORECASE), '{f}'),
    (re.compile(r'\{mi\}', re.IGNORECASE), '{f}'),
    (re.compile(r'\{munus\}', re.IGNORECASE), '{f}'),
    # Plural
    (re.compile(r'\{mesh\}', re.IGNORECASE), '{mesh}'),
    (re.compile(r'\{meš\}', re.IGNORECASE), '{mesh}'),
    # Wood/tree
    (re.compile(r'\{gish\}', re.IGNORECASE), '{gish}'),
    (re.compile(r'\{giš\}', re.IGNORECASE), '{gish}'),
    (re.compile(r'\{ĝeš\}', re.IGNORECASE), '{gish}'),
    (re.compile(r'\{geš\}', re.IGNORECASE), '{gish}'),
    # Reed
    (re.compile(r'\{gi\}', re.IGNORECASE), '{gi}'),
    # Stone
    (re.compile(r'\{na4\}', re.IGNORECASE), '{na4}'),
    (re.compile(r'\{na₄\}', re.IGNORECASE), '{na4}'),
    # Metal
    (re.compile(r'\{urudu\}', re.IGNORECASE), '{urudu}'),
    # Textile
    (re.compile(r'\{tug2\}', re.IGNORECASE), '{tug2}'),
    (re.compile(r'\{tug₂\}', re.IGNORECASE), '{tug2}'),
    (re.compile(r'\{túg\}', re.IGNORECASE), '{tug2}'),
    # City
    (re.compile(r'\{uru\}', re.IGNORECASE), '{uru}'),
    # Land/country
    (re.compile(r'\{kur\}', re.IGNORECASE), '{kur}'),
    # Stars
    (re.compile(r'\{mul\}', re.IGNORECASE), '{mul}'),
    # Buildings
    (re.compile(r'\{e₂\}', re.IGNORECASE), '{e2}'),
    (re.compile(r'\{é\}', re.IGNORECASE), '{e2}'),
    # Tablet/document
    (re.compile(r'\{dub\}', re.IGNORECASE), '{dub}'),
    # River/canal
    (re.compile(r'\{id₂\}', re.IGNORECASE), '{id2}'),
    (re.compile(r'\{íd\}', re.IGNORECASE), '{id2}'),
    # Birds
    (re.compile(r'\{mušen\}', re.IGNORECASE), '{mushen}'),
    # Skin/leather
    (re.compile(r'\{kuš\}', re.IGNORECASE), '{kush}'),
    # Plants
    (re.compile(r'\{u₂\}', re.IGNORECASE), '{u2}'),
    (re.compile(r'\{ú\}', re.IGNORECASE), '{u2}'),
    # Professions
    (re.compile(r'\{lu₂\}', re.IGNORECASE), '{lu2}'),
    (re.compile(r'\{lú\}', re.IGNORECASE), '{lu2}'),
]

# ── Accented vowel mappings (index notation → diacritics) ──────────────────
# a2→á, a3→à, e2→é, e3→è, i2→í, i3→ì, u2→ú, u3→ù
_VOWEL_ACCENT_PATTERNS = [
    (re.compile(r'a2\b'), 'á'),
    (re.compile(r'a3\b'), 'à'),
    (re.compile(r'e2\b'), 'é'),
    (re.compile(r'e3\b'), 'è'),
    (re.compile(r'i2\b'), 'í'),
    (re.compile(r'i3\b'), 'ì'),
    (re.compile(r'u2\b'), 'ú'),
    (re.compile(r'u3\b'), 'ù'),
    (re.compile(r'A2\b'), 'Á'),
    (re.compile(r'A3\b'), 'À'),
    (re.compile(r'E2\b'), 'É'),
    (re.compile(r'E3\b'), 'È'),
    (re.compile(r'I2\b'), 'Í'),
    (re.compile(r'I3\b'), 'Ì'),
    (re.compile(r'U2\b'), 'Ú'),
    (re.compile(r'U3\b'), 'Ù'),
]

# ── Roman numeral month mapping ────────────────────────────────────────────
_ROMAN_MONTHS = {
    'XII': '12', 'XI': '11', 'VIII': '8', 'VII': '7',
    'VI': '6', 'IV': '4', 'IX': '9', 'III': '3',
    'II': '2', 'XIII': '13', 'X': '10', 'V': '5', 'I': '1',
}

# ── Fraction mapping ──────────────────────────────────────────────────────
_FRACTION_MAP = {
    '1/2': '½', '1/3': '⅓', '2/3': '⅔',
    '1/4': '¼', '3/4': '¾', '1/5': '⅕',
    '1/6': '⅙', '1/8': '⅛',
}


def clean_transliteration(text: str) -> str:
    """Clean and normalize an Akkadian transliteration string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Convert Ḫ/ḫ → H/h (competition convention — test data uses only H/h)
    text = text.replace("Ḫ", "H").replace("ḫ", "h")

    # Remove Hamza/Ayin/glottal characters (ʾ ʿ ʼ etc.)
    for ch in _GLOTTAL_CHARS:
        text = text.replace(ch, "")

    # Convert subscript digits ₀-₉ → 0-9, ₓ → x
    text = text.translate(_SUBSCRIPT_MAP)

    # Normalize determinatives
    for pattern, replacement in _DETERMINATIVE_PATTERNS:
        text = pattern.sub(replacement, text)

    # Standardize gap markers: various forms → <gap> / <big_gap>
    # Large breaks first
    text = re.sub(r'\{large break\}', '<big_gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\{big gap\}', '<big_gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[…\s*…\]', '<big_gap>', text)
    text = re.sub(r'…', '<big_gap>', text)
    # Small breaks
    text = re.sub(r'\[(?:x\s*)+\]', '<gap>', text)
    text = re.sub(r'\[\.{2,}\]', '<gap>', text)
    text = re.sub(r'\.{3,}', '<gap>', text)
    text = re.sub(r'\[broken\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[damaged\]', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[missing\]', '<gap>', text, flags=re.IGNORECASE)

    # Remove scribal notations: keep text inside brackets/half-brackets
    # Half-brackets (uncertain reading): ˹text˺ → text
    text = re.sub(r'[˹⸢]', '', text)
    text = re.sub(r'[˺⸣]', '', text)
    # Double angle brackets (errant signs): <<text>> → remove entirely
    text = re.sub(r'<<[^>]*>>', '', text)
    # Square brackets (restoration): [text] → text (but NOT <gap>/<big_gap>)
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    # Angle brackets (scribal error correction): <text> → text (but NOT <gap>/<big_gap>)
    text = re.sub(r'<(?!gap>|big_gap>)([^>]*)>', r'\1', text)
    # Remove !, ?, *, # (editorial marks on individual signs)
    text = re.sub(r'[!?*#]', '', text)

    # Convert alternate romanization: sz→š, s,→ṣ, t,→ṭ
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r's,', 'ṣ', text)
    text = re.sub(r'S,', 'Ṣ', text)
    text = re.sub(r't,', 'ṭ', text)
    text = re.sub(r'T,', 'Ṭ', text)

    # Convert accented vowel notation: a2→á, a3→à, etc.
    for pattern, replacement in _VOWEL_ACCENT_PATTERNS:
        text = pattern.sub(replacement, text)

    # Deduplicate sequential gaps
    text = re.sub(r'(<big_gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    # Collapse mixed gap sequences to big_gap
    text = re.sub(r'(<(?:big_)?gap>\s*){2,}', '<big_gap> ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_translation(text: str) -> str:
    """Clean and normalize an English translation string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove scholarly annotations in parentheses
    # e.g., (fem. plur.), (sic), (lit. "..."), (var.), (obscure), etc.
    text = re.sub(
        r'\((?:fem\.?|masc\.?|plur\.?|sing\.?|sic|lit\.?|var\.?|obscure|'
        r'broken|damaged|illegible|erased|cf\.|i\.e\.|viz\.|sc\.|'
        r'fem\.\s*plur\.?|masc\.\s*plur\.?|fem\.\s*sing\.?|masc\.\s*sing\.?)'
        r'[^)]*\)',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Clean remaining editorial brackets
    # Square brackets: [restored text] → restored text
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    # Angle brackets
    text = re.sub(r'<(?!gap>)([^>]*)>', r'\1', text)
    # Half-brackets
    text = re.sub(r'[˹⸢˺⸣]', '', text)

    # Normalize gap markers in translations
    text = re.sub(r'\[…\s*…\]', '<big_gap>', text)
    text = re.sub(r'…', '<big_gap>', text)
    text = re.sub(r'\[\.{2,}\]', '<gap>', text)
    text = re.sub(r'\.{3,}', '<gap>', text)
    text = re.sub(r'\bbroken\b', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdamaged\b', '<gap>', text, flags=re.IGNORECASE)

    # Normalize fractions to Unicode
    for frac, uni in _FRACTION_MAP.items():
        text = text.replace(frac, uni)

    # Convert Roman numeral months to Arabic
    # Pattern: "Month XII" or "month IV" etc.
    def _replace_roman_month(m):
        prefix = m.group(1)
        roman = m.group(2)
        arabic = _ROMAN_MONTHS.get(roman, roman)
        return f"{prefix}{arabic}"

    text = re.sub(
        r'((?:[Mm]onth)\s+)(XIII|XII|XI|VIII|VII|VI|IX|IV|III|II|X|V|I)\b',
        _replace_roman_month,
        text
    )

    # Deduplicate sequential gaps
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test transliteration cleaning
    test_cases_translit = [
        "a-na {d}UTU ša₂ an-ni-i₃",
        "[x x x] lu₂-ša₂-lim",
        "˹ša₂˺ DINGIR-šu₂ ... i-na",
        "a-na sz-a-ri s,a-ab-tu t,up-pi",
    ]
    print("=== Transliteration Cleaning ===")
    for t in test_cases_translit:
        print(f"  IN:  {t}")
        print(f"  OUT: {clean_transliteration(t)}")
        print()

    # Test translation cleaning
    test_cases_trans = [
        "He said (lit. spoke) to the king.",
        "In Month XII, the offering [was made].",
        "The woman (fem. plur.) brought 1/2 mina of silver.",
        "... the tablet is damaged ...",
    ]
    print("=== Translation Cleaning ===")
    for t in test_cases_trans:
        print(f"  IN:  {t}")
        print(f"  OUT: {clean_translation(t)}")
        print()
