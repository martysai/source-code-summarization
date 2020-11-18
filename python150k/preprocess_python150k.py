import argparse
import ast
import asttokens
from io import BytesIO
import numpy as np
import os
import pickle
import re
import shutil
import tokenize
from typing import Tuple, List


SPECIAL_WORD = "JKMCFHNBVCXSDJ"
SPACE_STOPWORDS = [' ', '\t', '\r', '\n', '\v', '\f']
TOKENS_STOPWORDS = SPACE_STOPWORDS + ["utf-8"]
DOCSTRING_PREFIX = "###DCSTR### "


def set_script_arguments(parser):
    # Main arguments
    main_args = parser.add_argument_group("Main")
    main_args.add_argument("--dirname", type=str, default="data",
                           help="The directory to be processed.")

    # Runtime arguments
    runtime = parser.add_argument_group("Environment")
    runtime.add_argument("--verbose", type=bool, default=True,
                         help="The script verbosity.")
    return runtime


class CommentProcessor:
    """
    Stores every comment from a given code file
    self.comments: lineno -> comment string
    """
    def __init__(self):
        self.comments = {}

    def parse_comments(self, code: str):
        """
        Prepares comments and save into self.comments dict.
        ---
        code: string, represents python's code
        """
        for lineno, line in enumerate(code.split("\n")):
            if '#' in line:
                comment = line[line.find('#') + 1:]
                line = line[:line.find('#')]
                # TODO: rewrite search of multi-line comments
                quotes1 = line.count('"')
                quotes2 = line.count("'")
                comment = tiny_filter(comment)
                if len(comment) and quotes1 % 2 == 0 and quotes2 % 2 == 0:
                    self.comments[lineno] = comment


def read_file_to_string(filename: str) -> str:
    """
    Reading file contents into string object
    """
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s


def tiny_filter(code: str) -> str:
    """
    Filter string with regular expressions.
    """
    code = code.strip()
    code = re.sub('\t+', ' ', code)
    code = re.sub(' +', ' ', code)
    return code


def get_tokens(
        code: str,
        special_word: str = SPECIAL_WORD,
        remember_word: str = "",
        docstring: str = "") -> Tuple[list, int, list]:
    """
    code: str,
        Represents Python's file in string.
    special_word: str,
        Special index to be put in node.id ?
    remember_word: str,
        Function name node.id.
    docstring: str,
        Is taken from ast.get_docstring. Usually is None.
    ---
    Returns:
        tokens: List[str]
            List of tokens in the current function
        comments: List[str]
            List of retrieved comments
        special_idx: str
            An index of special word occurence.
        remember_idxs: List[str]
            A list of remember word occurences.
    """
    global TOKENS_STOPWORDS
    tokens = []
    remember_idxs = []
    special_idx = -1
    comment_ind = 0

    comments = []

    for idx, token in enumerate(tokenize.tokenize(
                      BytesIO(code.encode('utf-8')).readline)):
        # Form indices and tokens
        if token.string not in TOKENS_STOPWORDS:
            tokens.append(token.string)
            if token.string == special_word:
                special_idx = idx
            if token.string == remember_word:
                remember_idxs.append(idx)

        # Form comments
        if token.string == DOCSTRING_PREFIX:
            # special case -- docstring delimiter
            comments.append([comment_ind, DOCSTRING_PREFIX + " " + docstring])
            comment_ind += 1
        elif len(''.join(filter(str.isalpha, token.string))) > 0:
            # common case -- readable docstring
            comments.append([comment_ind, token.string])
            comment_ind += 1
    return tokens, comments, special_idx, remember_idxs


def get_previous_comments(
        fun: ast.FunctionDef,
        code_lines: List[str]) -> str:
    """
    Returns a comment on the line above the function definition.
    ---
    fun: ast.FunctionDef,
        Function object.
    code_lines: str,
        Special index to be put in node.id ?
    """
    fun_line_first = fun.first_token.start[0] - 1

    comment_line = code_lines[fun_line_first - 1].strip()
    zero_line = code_lines[fun_line_first - 2].strip()

    if comment_line[0] == "#":
        if (fun_line_first >= 2 and len(comment_line) >= 1 and
                zero_line == "") or (fun_line_first == 1 and
                                     len(comment_line) >= 1):
            precomment = code_lines[fun_line_first - 1].strip()
    return precomment


def collect_data(
        filename: str,
        args: argparse.ArgumentParser,
        special_word: str = SPECIAL_WORD) -> List[List[str]]:
    """
    Read an 2 unparallel corpuses: functions and docstrings.
    ---
    Returns:
        data: List[List[str]]
            Return summarized data from functions.
    """
    if args.verbose:
        print("Running collect_docstrings")
    code = read_file_to_string(filename)
    code_lines = code.splitlines()

    atok = asttokens.ASTTokens(code, parse=True)
    astree = atok.tree

    data = []

    # Global loop: iterating over functions from file
    for fun_ind, fun in enumerate(ast.iter_child_nodes(astree)):
        if isinstance(fun, ast.FunctionDef) and len(fun.body) > 0:
            fun_begin = fun.first_token.startpos
            fun_end = fun.last_token.endpos
            prev_comment = get_previous_comments(fun, code_lines)
            body_start = fun.body[0].first_token.startpos
            docstring = ast.get_docstring(fun)
            if args.verbose:
                print("Function ind:", fun_ind)
                print("len(fun.body):", len(fun.body))
                print("Docstring:", docstring)
            if not docstring:
                docstring = ""
            else:
                docstring = DOCSTRING_PREFIX + docstring + "\n"

            # Forming scope -- set of node ids (variables)
            scope = [arg.arg for arg in fun.args.args]
            for node in ast.walk(fun):
                if isinstance(node, ast.Name) and \
                   isinstance(node.ctx, ast.Store):
                    scope.append(node.id)
            scope = set(scope)

            if len(scope) < 3:
                print(f"Note: Function with fun.id = {fun.id} has too small"
                      "scope.")
            else:
                is_data_empty = True
                for node in ast.walk(fun):
                    if isinstance(node, ast.Name) and \
                        isinstance(node.ctx, ast.Load) and \
                            node.id in scope:

                        # Form current node contents to parse and find comments
                        id_ = node.id
                        startpos = node.first_token.startpos
                        endpos = node.last_token.endpos

                        # Special format for get_token
                        node_code = prev_comment
                        node_code += code[fun_begin:body_start]
                        node_code += docstring
                        node_code += code[body_start:startpos]
                        node_code += special_word
                        node_code += code[endpos:fun_end]

                        tokens, comments, idx, remember_idxs = \
                            get_tokens(node_code, special_word, id_)

                        # Checker: get_tokens returned proper output
                        assert tokens[idx] == special_word, \
                               f"{tokens[idx]} != {special_word}"

                        assert np.all([tokens[remem_idx] == id_
                                       for remem_idx in remember_idxs])

                        node.id = id_
                        if is_data_empty:
                            tokens[idx] = id_
                            data.append([code[fun_begin:fun_end],
                                         tokens,
                                         comments,
                                         []
                                         ])
                            is_data_empty = False
                        data[-1][-1].append([id_, idx, remember_idxs])

            # Delete rubbish object for this function if necessary
            if not is_data_empty and len(data[-1][-1]) <= 2:
                del data[-1]
        elif args.verbose:
            print(f"An object with fun_ind={fun_ind} is not a function.")
    return data


def retrieve_functions_docstrings(
        data: List,
        args: argparse.ArgumentParser) -> List[List[str]]:
    """
    add description
    ---
    Returns:
        comments: List[str],
            Functions comments separately.
        docstrings: List[str],
            Tokenized docstring corpus.
        functions: List[str],
            Tokenized function corpus.
        ord_nodes: List[*]
            Data consists of (node.id, token ind, remember ind)
            objects for further masking and processing.
        tokens: List[str],
            Functions tokens separately.
    """
    comments = []
    docstrings = []
    functions = []
    ord_nodes = []
    tokens = []

    for code, fun_tokens, fun_comments, _, ord_nodes_data in data:
        # Add asserts for debugging
        assert type(code) == str, "code variable is not a string"
        assert type(fun_tokens) == list, \
            "fun_tokens variable is not a list"
        assert type(fun_comments) == list, \
            "fun_comments variable is not a list"
        assert type(ord_nodes_data) == list, \
            "ord_nodes_data variable is not a string"

        functions.append(code)
        tokens.append(fun_tokens)
        comments.append(fun_comments)
        ord_nodes.append(ord_nodes_data)

        # Retrieve docstrings by predefined prefix
        fun_dcstrs = []
        for fun_cmt in fun_comments:
            comment_content = fun_cmt[1]
            docstring_ind = comment_content.find(DOCSTRING_PREFIX)
            if docstring_ind != -1:
                # find beginning of comment contents
                # + 1 is necessary because of space symbol
                begin = docstring_ind + len(DOCSTRING_PREFIX) + 1
                fun_dcstrs.append(comment_content[begin:])

        assert len(fun_dcstrs) <= 1, "Two or more docstrings were found."

        docstring = fun_dcstrs[0]
        docstrings.append(docstring)

    return comments, docstrings, functions, ord_nodes, tokens


def main(args):
    if os.path.exists(args.dirname):
        shutil.rmtree(args.dirname)

    comments_dir = "python150k_comments"
    docstrings_dir = "python150k_docstrings"
    functions_dir = "python150k_functions"
    ord_nodes_dir = "python150k_ord_nodes"
    tokens_dir = "python150k_tokens"

    for filename in os.listdir(args.dirname):
        data = collect_data(filename, args)
        comments, docstrings, functions, ord_nodes, tokens = \
            retrieve_functions_docstrings(data, args)

        comments_file = open(os.path.join(comments_dir,
                                          filename), "a")
        docstrings_file = open(os.path.join(docstrings_dir,
                                            filename), "a")
        functions_file = open(os.path.join(functions_dir,
                                           filename), "a")
        ord_nodes_file = open(os.path.join(ord_nodes_dir,
                                           filename), "a")
        tokens_file = open(os.path.join(tokens_dir,
                                        filename), "a")

        for comment in comments:
            comments_file.write(comment)
        for function in functions:
            functions_file.write(function)
        for docstring in docstrings:
            docstrings_file.write(docstring)
        for token in tokens:
            tokens_file.write(token)

        pickle.dump(ord_nodes, ord_nodes_file)

        comments_file.close()
        docstrings_file.close()
        functions_file.close()
        ord_nodes_file.close()
        tokens_file.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Python150k preprocessing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_script_arguments(parser)
    args, unknown = parser.parse_known_args()
    main(args)
