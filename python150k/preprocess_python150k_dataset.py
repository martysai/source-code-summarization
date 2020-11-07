import argparse
import ast
import astor
from io import BytesIO
import parse_python3
import re
import tokenize

def set_script_arguments(parser):
    # Runtime arguments
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--verbose', type=bool, default=True,
                         help='verbosity of this scripts')

class CommentProcessor:
    '''
    Stores every comment from a given code file
    self.comments: lineno -> comment string
    '''
    def __init__(self):
        self.comments = {}
    
    def parse_comments(self, code):
        '''
        code: string, represents python's code
        '''
        for lineno, line in enumerate(code.split("\n")):
            if "#" in line:
                comment = line[line.find("#") + 1:]
                line = line[:line.find("#")]
                # TODO: rewrite search of multi-line comments
                quotes1 = line.count('"')
                quotes2 = line.count("'")
                comment = re.sub('\t+', ' ', comment.strip().replace("#", ""))
                comment = re.sub(' +', ' ', comment)
                if len(comment) and quotes1 % 2 == 0 and quotes2 % 2 == 0:
                    self.comments[lineno] = comment


SPACE_STOPWORDS = [' ', '\t', '\r', '\n', '\v', '\f']
TOKENS_STOPWORDS = SPACE_STOPWORDS + ['utf-8']

# Returns tokens from default Python's tokenize
def get_tokens(f, special_word, remember_word):
    global TOKENS_STOPWORDS
    # Useful regular expressions for preprocessing
    code = re.sub('\t+', ' ', f.strip())
    code = re.sub(' +', ' ', code)
    tokens = []
    remember_idxs = []
    special_idx = -1
    for idx, token in enumerate(tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)):
        if token.string not in TOKENS_STOPWORDS:
            tokens.append(token.string)
            if token.string == special_word:
                special_idx = idx
            if token.string == remember_word:
                remember_idxs.append(idx)
    return tokens, special_idx, remember_idxs




def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Python150k preprocessing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    set_script_arguments(parser)
    args, unknown = parser.parse_known_args()
    main(args)