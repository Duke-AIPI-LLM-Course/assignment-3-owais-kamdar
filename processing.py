# processing.py

def extract_sections(cleaned_text, cleaned_toc):
    """
    Extracts sections from cleaned_text based on TOC headings.
    """
    sections = []
    remaining_text = cleaned_text

    for i, heading in enumerate(cleaned_toc):
        # find the start index of the heading
        start_idx = remaining_text.find(heading)
        if start_idx == -1:
            continue 

        # find the next heading to determine section boundaries
        end_idx = len(remaining_text)
        if i + 1 < len(cleaned_toc):
            next_start = remaining_text.find(cleaned_toc[i + 1])
            if next_start != -1:
                end_idx = next_start

        # extract section content
        section_content = remaining_text[start_idx + len(heading):end_idx].strip()
        sections.append({"heading": heading, "content": section_content})

        # update remaining text to start after the current heading
        remaining_text = remaining_text[end_idx:]

    return sections


def chunk_text(cleaned_text, cleaned_toc, max_chunk_size=800, overlap=100):
    """
    Chunks the cleaned text within each TOC section with overlap.
    """
    sections = extract_sections(cleaned_text, cleaned_toc)
    all_chunks = []

    for section in sections:
        heading, content = section["heading"], section["content"]
        current_chunk = ""
        chunk_id = 0

        # split content into sentences
        sentences = [s.strip() for s in content.split(". ") if s.strip()]

        for sentence in sentences:
            sentence += ". "
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    all_chunks.append({
                        "heading": heading,
                        "content": current_chunk.strip(),
                        "chunk_id": f"{heading}_{chunk_id}"
                    })
                    chunk_id += 1
                    current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""

                # long sentences that exceed chunk size
                while sentence:
                    available_space = max_chunk_size - len(current_chunk)
                    if len(sentence) <= available_space:
                        current_chunk += sentence
                        sentence = ""
                    else:
                        split_point = sentence[:available_space].rfind(". ")
                        if split_point == -1:
                            split_point = available_space
                        chunk_to_add = sentence[:split_point + 1] if split_point != -1 else sentence[:available_space]
                        current_chunk += chunk_to_add
                        all_chunks.append({
                            "heading": heading,
                            "content": current_chunk.strip(),
                            "chunk_id": f"{heading}_{chunk_id}"
                        })
                        chunk_id += 1
                        current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                        sentence = sentence[split_point + 1:].strip() if split_point != -1 else sentence[available_space:].strip()

        if current_chunk:
            all_chunks.append({
                "heading": heading,
                "content": current_chunk.strip(),
                "chunk_id": f"{heading}_{chunk_id}"
            })

    return all_chunks

