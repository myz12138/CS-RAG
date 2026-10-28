import re


def to_chunks(title2text, max_chars=600):
    chunks, titles = [], []
    for title, text in title2text.items():
        buffer = ""
        for paragraph in re.split(r"\n{2,}", text or ""):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if not buffer:
                buffer = paragraph
            elif len(buffer) + 2 + len(paragraph) <= max_chars:
                buffer += "\n\n" + paragraph
            else:
                chunks.append(buffer)
                titles.append(title)
                buffer = paragraph
        if buffer:
            chunks.append(buffer)
            titles.append(title)
    return chunks, titles
