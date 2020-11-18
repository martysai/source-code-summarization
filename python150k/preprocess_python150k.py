import argparse
import ast
import asttokens
from io import BytesIO
import re
import tokenize
from typing import Tuple, List


SPECIAL_WORD = "JKMCFHNBVCXSDJ"
SPACE_STOPWORDS = [' ', '\t', '\r', '\n', '\v', '\f']
TOKENS_STOPWORDS = SPACE_STOPWORDS + ['utf-8']
DOCSTRING_PREFIX = "###DCSTR### "


def set_script_arguments(parser):
    # Runtime arguments
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--verbose', type=bool, default=True,
                         help='verbosity of this scripts')
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
    Returns tokens from default Python's tokenize.
    ---
    code: str,
        Represents Python's file in string.
    special_word: str,
        Special index to be put in node.id ?
    remember_word: str,
        Function name node.id.
    docstring: str,
        Is taken from ast.get_docstring. Usually is None.
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
    return tokens, special_idx, remember_idxs


# Return comments before function
def get_pre_comments(
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


def collect_docstrings(
        filename,
        args):
    """
    Read functions descriptions from a file.
    """
    if args.verbose:
        print("Running collect_docstrings")
    code = read_file_to_string(filename)
    code_lines = code.splitlines()

    atok = asttokens.ASTTokens(code, parse=True)
    astree = atok.tree

    functions = []
    comments = []

    for fun_ind, fun in enumerate(ast.iter_child_nodes(astree)):
        if isinstance(fun, ast.FunctionDef) and len(fun.body) > 0:
            a_fun, b_fun = fun.first_token.startpos, fun.last_token.endpos
            precomment = get_pre_comments(fun, code_lines)
            body_start = fun.body[0].first_token.startpos
            docstring = ast.get_docstring(fun)
            if args.verbose:
                print("Function ind:", fun_ind, "; len(fun.body):", len(fun.body))
                print("Docstring:", docstring)
            if not docstring:
                docstring = ""
            else:
                docstring = DOCSTRING_PREFIX + docstring + "\n"
            scope = [arg.arg for arg in fun.args.args]
            for node in ast.walk(fun):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    scope.append(node.id)        
            scope = set(scope)
            if len(scope) < 3:
                print(f"Note: Function with fun.id = {fun.id} has too small scope.")
            else:
                first = True
                for node in ast.walk(fun):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in scope:
                        # Form current node contents to parse and find comments
                        id_ = node.id
                        a, b = node.first_token.startpos, node.last_token.endpos
                        node_code = precomment + code[a_fun:body_start] + docstring + \
                            code[body_start:a] + special_word + code[b:b_fun] + postcomment

                        tokens, idx, remember_idxs, comments = get_tokens(node_code, special_word, id_)
                        assert tokens[idx] == special_word, f"{tokens[idx]} != {special_word}"
                        assert np.all([tokens[remem_idx]==id_ for remem_idx in remember_idxs])

                        node.id = id_
                        if first:
                            tokens[idx] = id_
                            data.append([code[a_fun:b_fun], tokens, comments, scope, []])
                            first = False
                        data[-1][-1].append([id_, id_, idx, remember_idxs])
                        buggy_id = list(scope.difference({id_}))\
                                [np.random.randint(low=0, high=len(scope)-1)]
                        data[-1][-1].append([id_, buggy_id, idx, remember_idxs])
            if not first and len(data[-1][-1]) <= 2:
                del data[-1]
    return data


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Python150k preprocessing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_script_arguments(parser)
    args, unknown = parser.parse_known_args()
    main(args)
