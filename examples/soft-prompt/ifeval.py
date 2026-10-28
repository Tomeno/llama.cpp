"""Verifiers of the IFEval constraints as encoded in allenai/RLVR-IFeval: a ground truth {"func_name", args...}.

Ported from open-instruct (open_instruct/if_functions.py, Apache-2.0), with a few fixes: JSON may come in a code
fence, bullets are "* " / "- " lines (not bold text or dividers), highlights are non-empty *...* spans on one line,
sections are counted as in the IFEval reference (splitter + number).
The response language check needs langdetect and is not supported.
"""
import json
import re


def _count_ok(n: int, N: int, quantifier: str | None, around: int) -> bool:
    if quantifier == "at least":
        return n >= N
    if quantifier == "at most":
        return n <= N
    if quantifier == "around":
        return abs(n - N) <= around
    return False


def keywords(text, keyword_list):
    return all(k.lower() in text.lower() for k in keyword_list)


def keyword_frequency(text, word, N):
    return re.findall(r"\b\w+\b", text.lower()).count(word.lower()) == N


def forbidden_words(text, forbidden_words):
    return not any(w.lower() in text.lower() for w in forbidden_words)


def letter_frequency(text, letter, N):
    return text.count(letter) == N


def paragraph_count(text, N):
    parts = "\n".join(ln.strip() for ln in text.splitlines()).strip().split("* * *")
    return all(p.strip() for p in parts) and len(parts) == N


def word_constraint(text, N, quantifier):
    return _count_ok(len(text.split()), N, quantifier, max(round(N * 0.1), 1))


def sentence_constraint(text, N, quantifier):
    n = len(re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text.strip()))
    return _count_ok(n, N, quantifier, 1)


def paragraphs(text, N, first_word, i):
    parts = [p for p in re.split(r"\n\s*\n", text.strip())]
    return len(parts) == N and parts[i - 1].strip().lower().startswith(first_word.lower())


def postscript(text, postscript_marker):
    i = text.find(postscript_marker)
    return i >= 0 and len(text[i:].strip()) > len(postscript_marker)


def placeholders(text, N):
    return len(re.findall(r"\[(.*?)\]", text)) >= N


def bullet_points(text, N):
    return len(re.findall(r"^\s*[*-]\s", text, re.M)) == N


def title(text):
    return re.search(r"<<[^\n]+>>", text) is not None


def choice(text, options):
    return any(o in text for o in options)


def highlighted_sections(text, N):
    return len([m for m in re.findall(r"\*([^\n*]*)\*", text) if m.strip()]) >= N


def sections(text, N, section_splitter):
    # as the IFEval reference: the splitter followed by a number marks a section
    return len(re.findall(re.escape(section_splitter.strip()) + r"\s?\d+", text)) == N


def json_format(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        json.loads(t)
    except (ValueError, RecursionError):
        return False
    return True


def repeat_prompt(text, original_prompt):
    return text.strip().lower().startswith(original_prompt.strip().lower())


def two_responses(text):
    parts = text.split("******")
    return len(parts) == 2 and parts[0].strip() != parts[1].strip() and all(p.strip() for p in parts)


def uppercase(text):
    return text == text.upper()


def lowercase(text):
    return text == text.lower()


def frequency_capital_words(text, N, quantifier):
    return _count_ok(len(re.findall(r"\b[A-Z]+\b", text)), N, quantifier, max(round(N * 0.1), 1))


def end(text, end_phrase):
    return text.strip().endswith(end_phrase.strip())


def quotation(text):
    t = text.strip()
    return len(t) > 1 and t[0] == '"' and t[-1] == '"'


def no_commas(text):
    return "," not in text


FUNCS = {
    "verify_keywords": keywords,
    "verify_keyword_frequency": keyword_frequency,
    "validate_forbidden_words": forbidden_words,
    "verify_letter_frequency": letter_frequency,
    "verify_paragraph_count": paragraph_count,
    "validate_word_constraint": word_constraint,
    "verify_sentence_constraint": sentence_constraint,
    "validate_paragraphs": paragraphs,
    "verify_postscript": postscript,
    "validate_placeholders": placeholders,
    "verify_bullet_points": bullet_points,
    "validate_title": title,
    "validate_choice": choice,
    "validate_highlighted_sections": highlighted_sections,
    "validate_sections": sections,
    "validate_json_format": json_format,
    "validate_repeat_prompt": repeat_prompt,
    "validate_two_responses": two_responses,
    "validate_uppercase": uppercase,
    "validate_lowercase": lowercase,
    "validate_frequency_capital_words": frequency_capital_words,
    "validate_end": end,
    "validate_quotation": quotation,
    "validate_no_commas": no_commas,
}


def check(text: str, gt: dict) -> bool:
    """The response (after the thinking) meets the constraint gt; the args are the non-null fields of gt."""
    f = FUNCS[gt["func_name"]]
    args = {k: v for k, v in gt.items() if k != "func_name" and v is not None and k in f.__code__.co_varnames}
    return bool(f(text.strip(), **args))
