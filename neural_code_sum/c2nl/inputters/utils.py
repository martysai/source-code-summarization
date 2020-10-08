import logging
import random
import string
import json
from collections import Counter
from tqdm import tqdm

from c2nl.objects import Code, Summary
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines

logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(lang_id,
                     source,
                     source_tag,
                     target,
                     rel_matrix,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True,
                     split_tokens=False):
    code_tokens = source.split()
    code_type = []
    if source_tag is not None:
        code_type = source_tag.split()
        if len(code_tokens) != len(code_type):
            return None
    if rel_matrix is not None:
        if len(rel_matrix) != len(code_tokens):
            raise ValueError("len(rel_matrix) != len(code_tokens): %d %d" % \
                            len(rel_matrix), len(code_tokens))
        rel_matrix = [s.split() for s in rel_matrix]
    else:
        rel_matrix = []

    code_tokens = code_tokens[:max_src_len]
    code_type = code_type[:max_src_len]
    rel_matrix = rel_matrix[:max_src_len]
    
    if len(code_tokens) == 0:
        return None

    TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
        code_tag_type == 'subtoken' else AST_TYPE_MAP
    code = Code()
    code.text = source
    code.language = lang_id
    code.tokens = code_tokens
    if split_tokens:
        code.subtokens = [token.split("_") for token in code_tokens]
        #print([token.split("_") for token in code_tokens])
    code.type = code_type
    #code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
    #if code_tag_type != 'subtoken':
    #    code.mask = [1 if ct == 'N' else 0 for ct in code_type]

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None
        
    if rel_matrix != []:
        rm = Code()
        rm.tokens = rel_matrix

    example = dict()
    example['code'] = code
    example['summary'] = summary
    if rel_matrix != []:
        example["rel_matrix"] = rm
    return example


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    with open(filenames['src']) as f:
        sources = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['src']))]

    if filenames['tgt'] is not None:
        with open(filenames['tgt']) as f:
            targets = [line.strip() for line in
                       tqdm(f, total=count_file_lines(filenames['tgt']))]
    else:
        targets = [None] * len(sources)

    if filenames['src_tag'] is not None:
        with open(filenames['src_tag']) as f:
            source_tags = [line.strip() for line in
                           tqdm(f, total=count_file_lines(filenames['src_tag']))]
    else:
        source_tags = [None] * len(sources)
        
    if args.use_tree_relative_attn: #filenames["rel_matrix"] is not None:
        with open(filenames["rel_matrix"]) as f:
            rel_matrices = [json.loads(line) for line in
                           tqdm(f, total=count_file_lines(filenames["rel_matrix"]))]
    else:
        rel_matrices = [None] * len(sources)
        
    print(len(sources), len(source_tags), len(targets), len(rel_matrices))
    assert len(sources) == len(source_tags) == len(targets) == len(rel_matrices)

    examples = []
    for src, src_tag, tgt, rel_matrix in tqdm(zip(sources, source_tags, targets, \
                                                  rel_matrices),
                                  total=len(sources)):
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   src,
                                   src_tag,
                                   tgt,
                                   rel_matrix,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split,
                                   split_tokens=args.sum_over_subtokens)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None, num_spec_tokens=2, \
               attrname="tokens"):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            tokens = getattr(ex[field], attrname)
            if type(tokens[0]) != list:
                _insert(tokens)
            else:
                for elem in tokens:
                    _insert(elem)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - num_spec_tokens if dict_size and dict_size > num_spec_tokens else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    special_token="pad_unk", attrname="tokens"):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size, \
                       num_spec_tokens=len(special_tokens.split("_")),\
                       attrname=attrname):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             special_tokens="pad_unk", attrname="tokens"):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size, \
                       num_spec_tokens=len(special_tokens.split("_")),\
                       attrname=attrname)
    dictionary = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        special_tokens)
    return dictionary


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
