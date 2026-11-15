"""Technical title formatting; no semantic reclassification of drafts."""
import re

from .catalog import NAMES

MAX_TITLE_LENGTH = 50
BAD_ENDING = re.compile(r'(?:[,;:–—\-/\\(«".]|\b(?:и|или|для|по|с|со|в|на|под|без))$', re.I)
PAIRS = {')': '(', ']': '[', '»': '«'}


def valid_title(title):
    if not isinstance(title, str) or not title or title != title.strip() or len(title) > MAX_TITLE_LENGTH:
        return False
    if BAD_ENDING.search(title):
        return False
    stack = []
    for char in title:
        if char in ('(', '[', '«'):
            stack.append(char)
        elif char in PAIRS and (not stack or stack.pop() != PAIRS[char]):
            return False
    return not stack and title.count('"') % 2 == 0


def repair_title(title, category_id):
    """Keep a complete clause before an unfinished aside, otherwise use catalog name.

    Never truncate at a character limit: a half-list or half-sentence is not a title.
    The catalog fallback asserts no operation beyond the already selected category.
    """
    title = title.strip()
    if valid_title(title):
        return title
    for separator in ('(', '«', ':'):
        if separator in title:
            prefix = title.split(separator, 1)[0].strip()
            if valid_title(prefix):
                return prefix
    return NAMES[category_id]
