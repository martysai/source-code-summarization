"""AST-aware metrics for evaluating code summarization quality.

These metrics go beyond surface-level text comparison by parsing the source
code's AST and checking whether the generated docstring correctly references
structural elements (identifiers, control-flow patterns, parameters, etc.).

Uses src.ast_utils for AST parsing.
"""


def ast_node_recall(predictions: list[str], code: list[str]) -> float:
    """Fraction of key AST identifiers mentioned in the generated docstring.

    Parses each code snippet's AST, extracts meaningful identifiers
    (function names, variable names, class references), and checks what
    fraction of them appear in the corresponding generated docstring.
    """
    raise NotImplementedError


def control_flow_accuracy(predictions: list[str], code: list[str]) -> float:
    """Evaluate whether docstrings correctly identify control-flow patterns.

    Checks if the generated docstring mentions relevant control-flow
    constructs (loops, conditionals, exception handling, recursion) that
    are present in the source code's AST.
    """
    raise NotImplementedError


def parameter_accuracy(predictions: list[str], code: list[str]) -> float:
    """Check if generated docstrings correctly reference function parameters.

    Parses function signatures from the AST and verifies that parameter
    names mentioned in the docstring actually exist in the function signature.
    """
    raise NotImplementedError
