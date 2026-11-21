import pymupdf
from pathlib import Path
import pymupdf4llm as mu
import rapidocr
# file_path=Path(r"C:\Users\yashw\Desktop\RAG\Sankar Group – AI Automation Internship Assignment.pdf")
import re
file_path=r"C:\Users\yashw\Desktop\RAG\Sankar Group – AI Automation Internship Assignment.pdf"

#-----------------------------------------------------------------------------

def pdf_open(file_path):
    try: 
        pages = mu.to_markdown(        # pages is a list of dicts with keys: metadata, text
                        file_path,
                        page_chunks=True,
                        use_ocr=True
                        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise FileNotFoundError(f"Path '{file_path}' does not exist.")
    
    temp_storage=[]

    for index,page in enumerate(pages):
       metadata = page['metadata']
    # metadata={"page":metadata["page"],"title":metadata["title"],"author":metadata["author"]}
       text = page['text']
    #    text = clean_text(text)
       temp_storage.append({'metadata': metadata, 'text': text})

    
    
    return temp_storage   # return list[dict]

#-------------------------------------------------------------------------
import logging

logging.basicConfig(
    filename="pdf_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def check_pages(pages):
    for page in pages:
        page_number = page.get("metadata", {}).get("page_number", "Unknown")
        text = page.get("text", "")

        if not text.strip():
            logging.warning(f"Page {page_number}: Empty text")

        elif len(text.strip()) < 20:
            logging.warning(f"Page {page_number}: Very little text")

        else:
            logging.info(f"Page {page_number}: Processed successfully")

    return pages

#-------------------------- removes header and footer from the text of each page --------------------------
from collections import Counter

def remove_repeated_lines(pages, min_occurrences=2):

    line_counter = Counter()

    if not pages:
        raise ValueError("The 'pages' list is empty. Cannot remove repeated lines.")

    # Step 1: Count repeated lines across pages
    for page in pages:
        lines = set(
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        )
        line_counter.update(lines)

    # Step 2: Identify repeated lines
    repeated_lines = {
        line
        for line, count in line_counter.items()
        if count >= min_occurrences
    }

    # Step 3: Modify only the text field
    for page in pages:
        lines = page["text"].splitlines()

        page["text"] = "\n".join(
            line for line in lines
            if line.strip() not in repeated_lines
        )

    return pages


# Usage

#-------------------------- cleans the text of each page --------------------------

def clean_text(pages):
    for page in pages:
        text = page["text"]

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Normalize spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading and trailing whitespace
        text = text.strip()

        page["text"] = text

    return pages

#------------------------------------------------------------------------
def validate_pages(pages):

    for page in pages:

        text = page["text"]
        page_number = page["metadata"]["page_number"]

        if not text.strip():
            print(f"Page {page_number}: Empty page")

        elif len(text.strip()) < 20:
            print(f"Page {page_number}: Very little text")

        elif "\ufffd" in text:
            print(f"Page {page_number}: Encoding issue")

        else:
            print(f"Page {page_number}: Looks OK")

    return pages
#-------------------------------------------------------------------------
def inspect_structure(pages):

    for page in pages:

        text = page["text"]
        page_number = page["metadata"]["page_number"]

        headings = [
            line for line in text.splitlines()
            if line.strip().startswith("#")
        ]

        tables = [
            line for line in text.splitlines()
            if "|" in line
        ]

        print(f"\nPage: {page_number}")
        print("Headings:", headings)
        print("Possible table lines:", len(tables))
#-----------------------------------------------------------------------------------
def validate_metadata(pages):

    for page in pages:

        metadata = page["metadata"]

        required_fields = [
            "page_number"
        ]

        for field in required_fields:
            if field not in metadata:
                print(f"Missing {field}")
            else:
                print(
                    f"Page {metadata['page_number']}: "
                    f"{field} = {metadata[field]}"
                )

    return pages
#---------------------------------------------------------------------
def validate_page_order(pages):
    page_numbers = [
        page["metadata"]["page_number"]
        for page in pages
    ]

    expected = sorted(page_numbers)

    if page_numbers == expected:
        print("Page order: Correct")
    else:
        print("Page order: Incorrect")
        print("Current order:", page_numbers)
        print("Expected order:", expected)

    return pages
#---------------------------------------------------------------------------
def review_extracted_text(pages):
    for page in pages:
        page_number = page["metadata"]["page_number"]

        print(f"\n{'=' * 50}")
        print(f"PAGE {page_number}")
        print("=" * 50)

        print(page["text"][:1000])
#--------------------------------------------------------------------------------
def final_function(file_path):
    final_output=pdf_open(file_path)
    final_output=check_pages(final_output)
    final_output=clean_text(final_output)
    final_output=remove_repeated_lines(final_output)
    final_output=validate_pages(final_output)
    final_output=validate_metadata(final_output)
    final_output=validate_page_order(final_output)
     

    return final_output





#----------------------------------------------------------------

output=final_function(file_path)



print(output[0])




    
