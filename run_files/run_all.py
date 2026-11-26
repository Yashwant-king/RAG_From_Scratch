import json
from pathlib import Path

from file_handling.csv.open_csv import final_function as csv_function
from file_handling.json.open_json import final_function as json_function
from file_handling.xlsx.open_xlsx import final_function as xlsx_function
from file_handling.txt.open_txt import final_function as txt_function
from file_handling.md.open_md import final_function as md_function
from file_handling.pdf.open_pdf import final_function as pdf_function

INPUT_DIR = Path(r"C:\Users\yashw\Desktop\RAG\input_files")
OUTPUT_FILE = Path(__file__).parent / "run_files" / "result.json"

HANDLERS = {
    ".csv": csv_function,
    ".json": json_function,
    ".xlsx": xlsx_function,
    ".txt": txt_function,
    ".md": md_function,
    ".pdf": pdf_function,
}


def main():
    results, skipped, failed = [], [], []

    for file in sorted(INPUT_DIR.iterdir()):  # use rglob("*") to include subfolders
        if not file.is_file():
            continue

        handler = HANDLERS.get(file.suffix.lower())
        if handler is None:
            skipped.append(file.name)
            continue

        try:
            results.append(handler(file))
        except Exception as e:
            failed.append((file.name, str(e)))
            print(f"[FAILED] {file.name}: {e}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=str)

    print(f"Processed: {len(results)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    if skipped:
        print("Skipped (unsupported):", skipped)


if __name__ == "__main__":
    main()