#extractor.py

import fitz
import re

def extract_data(pdf_path):
    """
    Extracts metadata, table of contents, and main content from the PDF.
    """
    doc = fitz.open(pdf_path)

    # extract metadata
    metadata = doc.metadata

    # extract from first page
    first_page = doc[0].get_text("text").strip()
    words = first_page.split()
    # get title and author
    title = " ".join(words[:6]) if len(words) > 6 else "Unknown Title"
    author = " ".join(words[6:]) if len(words) > 6 else "Unknown Author"
    metadata['title'] = title
    metadata['author'] = author

    # extract toc
    toc_start, toc_end = 3, 5
    table_of_contents = "\n".join([doc[page_num].get_text() for page_num in range(toc_start, toc_end)])

    # main content
    start_page = 6
    structured_content = []

    for page_num in range(start_page, len(doc)):
        page_text = doc[page_num].get_text("text")
        structured_content.append("\n" + page_text.strip() + "\n")

    raw_main_content = "\n".join(structured_content)

    return metadata, table_of_contents, raw_main_content


def clean_and_organize_text(text):
    """
    Cleans text by removing unwanted headers, fixing formatting issues, 
    and normalizing structure.
    """
    # split text into lines and filter out lines that are page or headers
    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # skip if the line is a roman numeral or a simple number
        if re.fullmatch(r'(?:[IVXLCDM]+|\d{1,2})', stripped, flags=re.IGNORECASE):
            continue
        # skip header phrase
        if re.search(r'\bDepression: An information guide\b', stripped, flags=re.IGNORECASE):
            continue
        filtered_lines.append(stripped)
    
    # rejoin the filtered lines into a single text block
    text = "\n".join(filtered_lines)
    # remove page numbers
    text = re.sub(r'\n\d+\n', ' ', text)
    # hyphenated words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # replace single newline characters with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)
    # punctuation
    text = re.sub(r'\s([?.!,"])', r'\1', text)
    # truncate the text at Glossary
    glossary_index = text.find("Glossary")
    if glossary_index != -1:
        text = text[:glossary_index]
    
    return text.strip()


def clean_toc(toc_text):
    """
    Cleans and organizes the Table of Contents (TOC).
    """
    # split toc into lines
    lines = [line.strip() for line in toc_text.split("\n") if line.strip()]
    # remove unwanted sections
    excluded_sections = {"Contents", "Acknowledgments", "Glossary", "Resources", "v", "vi"}
    filtered_toc = [section for section in lines if section not in excluded_sections]
    # remove page numbers
    filtered_toc = [line for line in filtered_toc if not re.fullmatch(r'\d+', line)]
    
    # split titles
    refined_toc = []
    i = 0

    while i < len(filtered_toc):
        if (
            i < len(filtered_toc) - 1 and 
            filtered_toc[i+1][0].islower()
        ):
            # merge headings
            merged_title = f"{filtered_toc[i]} {filtered_toc[i+1]}".strip()
            refined_toc.append(merged_title)
            i += 2 
        else:
            refined_toc.append(filtered_toc[i])
            i += 1

    return refined_toc


