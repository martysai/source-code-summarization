# Python Method Dataset

Source: [A Transformer-based Approach for Source Code Summarization](https://arxiv.org/abs/2005.00653) (ACL 2020)

Original repository: [wasiahmad/NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum)

## Dataset Statistics

| Split | Examples |
|-------|----------|
| Train | 55,538 |
| Dev | 18,505 |
| Test | 18,502 |
| Total | 92,545 |

## Format

Each split contains parallel files:
- `code.original` - Space-separated code tokens (one function per line)
- `code.original_subtoken` - Subtoken-split version (camelCase aware)
- `javadoc.original` - Space-separated summary tokens (one docstring per line)

## Download

Run `get_data.sh` to download and extract the dataset from Google Drive.
