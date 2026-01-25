"""Convert C2NL parallel-file dataset to HuggingFace instruction-tuning format.

The original dataset (from NeuralCodeSum / ACL 2020) stores code and summaries as
parallel text files with space-separated tokens. This script reads those files
(from zip archives), applies basic detokenization, and outputs a HuggingFace
DatasetDict suitable for instruction-tuning a code LLM.

Input format (inside zip archives):
    {split}/code.original       - Space-separated code tokens, one function per line
    {split}/javadoc.original    - Space-separated summary tokens, one docstring per line

Output format (HuggingFace Dataset):
    Columns: code, summary, instruction, response

Usage:
    python -m src.data.convert_seed \
        --input-dir data/raw/python-method \
        --output-dir data/processed/python-method
"""

import argparse
import zipfile
from io import TextIOWrapper
from pathlib import Path

from datasets import Dataset, DatasetDict

INSTRUCTION_TEMPLATE = (
    "Generate a concise docstring for the following Python function:\n\n{code}"
)

SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


def detokenize_code(token_line: str) -> str:
    """Reconstruct readable code from space-separated tokens.

    The C2NL code.original format strips most Python syntax (no colons, no
    parentheses around arguments, no indentation). This function applies
    heuristic rules to produce more readable code, though perfect reconstruction
    is not possible from the tokenized format.

    The result is "readable enough" for an LLM to understand the function's
    intent, which is sufficient for seed data that will be expanded via
    distilabel with properly-formatted code.
    """
    tokens = token_line.strip().split()
    if not tokens:
        return ""

    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Add space before token (with exceptions)
        if result and token not in (".", ",", ")", "]", "}") and result[-1] not in ("(", "[", "{", "."):
            result.append(" ")

        result.append(token)
        i += 1

    return "".join(result)


def detokenize_summary(token_line: str) -> str:
    """Reconstruct a readable summary sentence from space-separated tokens."""
    tokens = token_line.strip().split()
    if not tokens:
        return ""

    # Capitalize first token, join with spaces
    text = " ".join(tokens)
    if text:
        text = text[0].upper() + text[1:]
    # Ensure ends with period
    if text and not text.endswith((".","!","?")):
        text += "."
    return text


def convert_split(input_dir: Path, zip_name: str, split_folder: str) -> Dataset:
    """Convert a single split from a zip archive to a HuggingFace Dataset.

    Args:
        input_dir: Directory containing the zip files.
        zip_name: Name of the zip file (e.g., "train.zip").
        split_folder: Folder name inside the zip (e.g., "train", "dev", "test").

    Returns:
        A HuggingFace Dataset with columns: code, summary, instruction, response.
    """
    zip_path = input_dir / zip_name

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {zip_path}. "
            f"Run data/raw/python-method/get_data.sh to download it."
        )

    records = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        code_path = f"{split_folder}/code.original"
        summary_path = f"{split_folder}/javadoc.original"

        with zf.open(code_path) as code_f, zf.open(summary_path) as summary_f:
            code_reader = TextIOWrapper(code_f, encoding="utf-8")
            summary_reader = TextIOWrapper(summary_f, encoding="utf-8")

            for code_line, summary_line in zip(code_reader, summary_reader):
                code = detokenize_code(code_line)
                summary = detokenize_summary(summary_line)

                if not code or not summary:
                    continue

                records.append({
                    "code": code,
                    "summary": summary,
                    "instruction": INSTRUCTION_TEMPLATE.format(code=code),
                    "response": summary,
                })

    return Dataset.from_list(records)


def convert_dataset(input_dir: Path, output_dir: Path) -> DatasetDict:
    """Convert all splits of the C2NL dataset to a HuggingFace DatasetDict.

    Args:
        input_dir: Directory containing train.zip, dev.zip, test.zip.
        output_dir: Where to save the HuggingFace dataset.

    Returns:
        The converted DatasetDict.
    """
    ds_dict = {}
    for hf_split, folder_name in SPLIT_MAP.items():
        zip_name = f"{folder_name}.zip"
        print(f"Converting {hf_split} split from {zip_name}...")
        ds_dict[hf_split] = convert_split(input_dir, zip_name, folder_name)
        print(f"  {len(ds_dict[hf_split])} examples")

    dataset = DatasetDict(ds_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    print(f"\nDataset saved to {output_dir}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert C2NL parallel-file dataset to HuggingFace format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/python-method"),
        help="Directory containing train.zip, dev.zip, test.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/python-method"),
        help="Where to save the HuggingFace dataset",
    )
    args = parser.parse_args()

    convert_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
