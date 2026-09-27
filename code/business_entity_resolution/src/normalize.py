"""Text normalization utilities for business names and addresses.

All logic here is purely local (regex / lookup tables baked into this file) —
no network calls, no external services. The Devanagari transliteration map is
a standard linguistic romanization table (not a business-data lookup).
"""
import re
import unicodedata

from us_in_states import US_STATES, INDIA_STATES

# Multi-word state/UT names -> single canonical token (applied before tokenizing
# so "new york" / "New  York" collapse the same way "ny" does).
_STATE_PHRASES = {
    "new york": "newyork", "new jersey": "newjersey", "new mexico": "newmexico",
    "new hampshire": "newhampshire", "north carolina": "northcarolina",
    "south carolina": "southcarolina", "north dakota": "northdakota",
    "south dakota": "southdakota", "rhode island": "rhodeisland",
    "west virginia": "westvirginia", "district of columbia": "districtofcolumbia",
    "andhra pradesh": "andhrapradesh", "arunachal pradesh": "arunachalpradesh",
    "himachal pradesh": "himachalpradesh", "madhya pradesh": "madhyapradesh",
    "tamil nadu": "tamilnadu", "uttar pradesh": "uttarpradesh",
    "west bengal": "westbengal", "jammu and kashmir": "jammukashmir",
    "jammu kashmir": "jammukashmir", "new delhi": "delhi",
}
_STATE_PHRASE_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(_STATE_PHRASES, key=len, reverse=True))
)

# Business legal-suffix / connector synonyms -> canonical short token
BUSINESS_SYNONYMS = {
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "llp": "llp", "llc": "llc",
    "company": "co", "co": "co",
    "and": "and", "&": "and",
    "enterprises": "ent", "enterprise": "ent",
    "industries": "ind", "industry": "ind",
    "associates": "assoc", "association": "assoc",
    "group": "grp", "services": "svc", "service": "svc",
    "solutions": "sol", "solution": "sol",
    "international": "intl", "intl": "intl",
    "technologies": "tech", "technology": "tech", "technologys": "tech",
    # French legal forms / connectors (France appears only in the test set)
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "societe": "co", "ste": "co", "compagnie": "co", "cie": "co",
    "et": "and", "etablissements": "ent", "ets": "ent",
    # Indian name variants
    "pvtltd": "pvt ltd", "limted": "ltd",
}

# Tokens that carry almost no discriminative power for blocking keys once the
# business-suffix vocabulary is normalized (still kept in similarity features).
NAME_STOPWORDS = {
    "inc", "corp", "ltd", "pvt", "llp", "llc", "co", "and", "ent", "ind",
    "assoc", "grp", "svc", "sol", "intl", "the", "of", "a", "an", "for",
    "sarl", "sas", "eurl", "sa", "sci", "le", "la", "les", "de", "du", "des",
}

# Address abbreviation synonyms -> canonical token
ADDRESS_SYNONYMS = {
    "street": "st", "st": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd",
    "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl",
    "suite": "ste", "ste": "ste",
    "unit": "unit",
    "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "near": "near", "opposite": "opp", "opp": "opp",
    "number": "no", "no": "no",
    "post": "po", "box": "box",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "saint": "st",
    # French street types
    "rue": "rue", "avenue": "ave", "av": "ave", "bd": "blvd", "boulevard": "blvd",
    "chemin": "ch", "ch": "ch", "allee": "all", "impasse": "imp", "imp": "imp",
    "route": "rte", "rte": "rte", "quai": "quai", "cours": "crs",
    "batiment": "bldg", "bat": "bldg", "etage": "fl",
    # Indian address terms
    "marg": "rd", "sadak": "rd",
    "nagar": "nagar", "ngr": "nagar", "colony": "col", "col": "col",
    "sector": "sec", "sec": "sec", "phase": "ph", "ph": "ph",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "expressway": "expy", "square": "sq", "sq": "sq", "terrace": "ter",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "heights": "hts",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
}

ADDRESS_STOPWORDS = {
    "st", "rd", "ave", "dr", "ln", "blvd", "ct", "cir", "pl", "ste", "unit",
    "apt", "fl", "bldg", "near", "opp", "no", "po", "box", "n", "s", "e",
    "w", "the", "of", "null", "na", "ch", "all", "imp", "rte", "crs",
    "sec", "ph", "col", "hwy", "pkwy", "expy", "sq", "ter", "de", "du",
    "des", "la", "le", "les", "cedex", "rue", "imp", "quai",
}

# Minimal Devanagari -> Latin romanization (ISO-15919-ish, simplified),
# covering common consonants, independent vowels and vowel signs (matras).
# A static linguistic table, not a lookup against any business or
# government database.
_DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ः": "h",
    "ँ": "n", "्": "",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y", "ऱ": "r", "ऴ": "l",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "।": " ", "॥": " ",
}


def transliterate_devanagari(text: str) -> str:
    """Best-effort romanization of Devanagari characters; passes through everything else."""
    out = []
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            out.append(_DEVANAGARI_MAP.get(ch, ""))
        else:
            out.append(ch)
    return "".join(out)


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    if not text:
        return ""
    text = transliterate_devanagari(text)
    text = strip_accents(text)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(text: str) -> str:
    """Return normalized name string (suffix synonyms applied, ready for tokenizing)."""
    cleaned = basic_clean(text)
    if not cleaned:
        return ""
    tokens = [BUSINESS_SYNONYMS.get(t, t) for t in cleaned.split(" ")]
    return " ".join(tokens)


def name_tokens(text: str, drop_stopwords: bool = True) -> list:
    norm = normalize_name(text)
    if not norm:
        return []
    toks = norm.split(" ")
    if drop_stopwords:
        toks = [t for t in toks if t and t not in NAME_STOPWORDS]
    else:
        toks = [t for t in toks if t]
    return toks


def _state_abbr_table(country: str) -> dict:
    c = (country or "").strip().lower()
    if c == "us":
        return US_STATES
    if c == "india":
        return INDIA_STATES
    return {}


def normalize_address(text: str, country: str = "") -> str:
    cleaned = basic_clean(text)
    if not cleaned:
        return ""
    cleaned = _STATE_PHRASE_RE.sub(lambda m: _STATE_PHRASES[m.group(0)], cleaned)
    state_abbr = _state_abbr_table(country)
    tokens = []
    for t in cleaned.split(" "):
        if t in state_abbr:
            tokens.append(state_abbr[t])
        else:
            tokens.append(ADDRESS_SYNONYMS.get(t, t))
    return " ".join(tokens)


def address_tokens(text: str, country: str = "", drop_stopwords: bool = True) -> list:
    norm = normalize_address(text, country)
    if not norm:
        return []
    toks = norm.split(" ")
    if drop_stopwords:
        toks = [t for t in toks if t and t not in ADDRESS_STOPWORDS and not t.isdigit()]
    else:
        toks = [t for t in toks if t]
    return toks


def address_numbers(text: str) -> list:
    """All digit-runs found in the raw address (house/building/PIN numbers)."""
    cleaned = basic_clean(text)
    return re.findall(r"\d+", cleaned)


def soundex(word: str) -> str:
    """Classic Soundex code, used only as a typo-tolerant blocking key."""
    if not word:
        return ""
    word = word.upper()
    codes = {
        **{c: "1" for c in "BFPV"},
        **{c: "2" for c in "CGJKQSXZ"},
        **{c: "3" for c in "DT"},
        "L": "4",
        **{c: "5" for c in "MN"},
        "R": "6",
    }
    first = word[0]
    tail = []
    prev = codes.get(first, "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            tail.append(code)
        if ch not in "HW":
            prev = code
    return (first + "".join(tail) + "000")[:4]
