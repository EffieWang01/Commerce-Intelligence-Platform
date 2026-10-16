"""Conservative keyboard product-head matching for the local search baseline."""

import re

from .query_rules import EXCLUSION


KEYBOARD = re.compile(r"\bkeyboards?\b|键盘", re.I)
KEYCAP = re.compile(r"\bkey\s*caps?\b|键帽", re.I)
MECHANICAL = re.compile(r"\bmechanical\b(?![-\s]+(?:feeling|feel|like|style)\b)|机械键盘", re.I)
COMPATIBILITY = re.compile(r"\b(?:for|fits?|compatible\s+with|works\s+with)\b", re.I)
ACCESSORIES = [
    ("keycap_tool", re.compile(r"\bkey\s*cap\s+(?:puller|removal|remover)\b|拔键器", re.I)),
    ("switch_tool", re.compile(r"\bswitch\s+(?:opener|tester|puller|remover)\b|开轴器|拔轴器", re.I)),
    ("keycaps", KEYCAP),
    ("cable", re.compile(r"\bcables?\b|连接线|数据线", re.I)),
    ("wrist_rest", re.compile(r"\b(?:wrist|palm)\s+(?:rest|pad)s?\b|手托|腕托", re.I)),
    ("stabilizer", re.compile(r"\bstabili[sz]ers?\b|卫星轴", re.I)),
    ("switches", re.compile(r"\b(?:keyboard\s+switch(?:es)?(?!\s+(?:opener|tester|puller|remover))|switch(?:es)?\s+(?:pack|set|kit))\b|轴体", re.I)),
]
ACCESSORY_HEAD = re.compile(
    r"\bkeyboards?\s+(?:(?:coiled|usb[- ]?c)\s+)?"
    r"(?:cables?|key\s*caps?|stabili[sz]ers?|switch(?:es)?|(?:wrist|palm)\s+(?:rest|pad)s?)\b",
    re.I,
)
NO_KEYBOARD = re.compile(r"\b(?:no\s+keyboard\s+included|keyboard\s+not\s+included|only\s+keycaps?)\b", re.I)
TARGET_LABELS = {
    "keyboard": "键盘本体", "mechanical_keyboard": "机械键盘本体", "keycaps": "键帽", "cable": "键盘线材",
    "wrist_rest": "键盘手托", "stabilizer": "卫星轴", "switches": "键盘轴体",
    "keycap_tool": "拔键工具", "switch_tool": "轴体工具",
}


def query_product_target(query: str) -> str | None:
    positive_query = EXCLUSION.sub(" ", query)
    if not (KEYBOARD.search(positive_query) or KEYCAP.search(positive_query)):
        return None
    for kind, pattern in ACCESSORIES:
        if pattern.search(positive_query):
            return kind
    return "mechanical_keyboard" if MECHANICAL.search(positive_query) else "keyboard"


def keyboard_product_type(title: str, description: str = "") -> str | None:
    if not (KEYBOARD.search(title) or KEYCAP.search(title)):
        return None
    head = COMPATIBILITY.split(title, maxsplit=1)[0]
    keyboard = KEYBOARD.search(head)
    matches = [(kind, match) for kind, pattern in ACCESSORIES if (match := pattern.search(head))]
    # The sold object normally precedes its compatibility list; bundled keycaps/cables
    # after "keyboard with ..." must not turn a complete keyboard into an accessory.
    if matches:
        kind, match = min(matches, key=lambda entry: entry[1].start())
        if keyboard is None or match.start() <= keyboard.start() or ACCESSORY_HEAD.search(head):
            return kind
        if NO_KEYBOARD.search(title + " " + description):
            return kind
    if NO_KEYBOARD.search(title + " " + description):
        return "parts"
    if keyboard:
        return "mechanical_keyboard" if MECHANICAL.search(head) else "keyboard"
    return None


def matches_product_target(target: str, title: str, description: str = "") -> bool:
    product_type = keyboard_product_type(title, description)
    if target == "keyboard":
        return product_type in {"keyboard", "mechanical_keyboard"}
    return product_type == target
