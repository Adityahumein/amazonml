"""Static US / India state abbreviation <-> full-name tables.

These are standard postal/geographic reference tables (like the street-type
abbreviation dictionary), not a lookup against any external business or
government database, so they are fine to bake in for address normalization.
"""

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "newhampshire", "nj": "newjersey", "nm": "newmexico", "ny": "newyork",
    "nc": "northcarolina", "nd": "northdakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhodeisland", "sc": "southcarolina",
    "sd": "southdakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "westvirginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}

INDIA_STATES = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachalpradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhyapradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya",
    "mz": "mizoram", "nl": "nagaland", "or": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu", "tg": "telangana",
    "tr": "tripura", "up": "uttarpradesh", "uk": "uttarakhand", "wb": "westbengal",
    "dl": "delhi", "jk": "jammukashmir", "ld": "lakshadweep", "py": "puducherry",
    "ch": "chandigarh",
}
