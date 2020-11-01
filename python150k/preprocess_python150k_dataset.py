import ast
import json as json
import os
import numpy as np
import re
import tokenize
from io import BytesIO
import asttokens
import traceback
from p_tqdm import p_map

# --- Define constants ---
FOLDER = "data_3_full_v1"
NEW_FOLDER = "processed_3_full_v1"
ERR_FOLDER = "err_logs_processed_full_v1"
NUM_PROC = 16

SPECIAL_WORD = "JKMCFHNBVCXSDJ"
DOCSTRING_WORD ="###DCSTR###"


def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s


def get_tokens(f, special_word, remember_word, docstring=""):
    code = f
    tokens = []
    remember_idxs = []
    special_idx = -1
    comments = []
    prev_string = None
    for token in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline):
        if not token.type in {55}:
            if token.type == 3:
                string_ = '"str"'
            elif token.type == 5:
                string_ = "<indent>"
            elif token.type == 6:
                string_ = "<dedent>"
            elif token.type == 0:
                string_ = "<endf>"
            elif token.type == 54:
                string_ = "<errtoken>"
            elif token.type == 56:
                string_ = "<nl>"
            elif token.type == 4: # to avoid "\r\n" and ""
                string_ = "\n"
            else:
                string_ = token.string
            if string_ not in {"\n", "<nl>"} or \
                (string_ == "\n" and prev_string != "\n") or\
                (string_ == "<nl>" and prev_string != "<nl>"):
                # filter out what we do not remember: repeating \n
                if string_ == "":
                    raise ValueError("%s\n%s %s %s"%(code, \
                                                        token.type, token.string, prev_string))
                tokens.append(string_)
                if string_ == special_word:
                    special_idx = len(tokens) - 2
                if string_ == remember_word:
                    remember_idxs.append(len(tokens) - 2)
            prev_string = string_
        else:
            # comment
            if token.string == docstring_word:
                comments.append([len(tokens) - 2, \
                                 docstring_word + " " + docstring])
            else:
                if len(''.join(filter(str.isalpha, token.string))) > 0:
                     comments.append([len(tokens) - 2, token.string])
    return tokens[1:], special_idx, remember_idxs, comments

def get_pre_post_comments(fun, code_lines):
    fun_line_first = fun.first_token.start[0] - 1
    fun_line_last = fun.last_token.end[0] - 1
    if (fun_line_first >= 2 and len(code_lines[fun_line_first - 1].strip()) >= 1 and \
        code_lines[fun_line_first - 1].strip()[0] == "#" \
        and code_lines[fun_line_first-2].strip() == "") or \
        (fun_line_first == 1 and len(code_lines[fun_line_first - 1].strip()) >= 1 and \
        code_lines[fun_line_first - 1].strip()[0] == "#"):
        precomment = code_lines[fun_line_first - 1].strip() + "\n"
    else:
        precomment = ""
    if (fun_line_last <= len(code_lines) - 3 and len(code_lines[fun_line_last + 1].strip()) >= 1 and \
        code_lines[fun_line_last + 1].strip()[0] == "#" \
        and code_lines[fun_line_last + 2].strip() == "") or\
        (fun_line_last == len(code_lines) - 2 and len(code_lines[fun_line_last + 1].strip()) >= 1 and \
         code_lines[fun_line_last + 1].strip()[0] == "#"):
        postcomment = "\n" + code_lines[fun_line_last + 1].strip()
    else:
        postcomment = ""
    return precomment, postcomment

def process_file(filenm, data):

    code = read_file_to_string(filenm)
    code_lines = code.splitlines()
    atok = asttokens.ASTTokens(code, parse=True)
    astree = atok.tree


    for fun in ast.iter_child_nodes(astree):
        if isinstance(fun, ast.FunctionDef) and len(fun.body) > 0:
            a_fun, b_fun = fun.first_token.startpos, fun.last_token.endpos
            precomment, postcomment = get_pre_post_comments(fun, code_lines)
            body_start = fun.body[0].first_token.startpos
            docstring = ast.get_docstring(fun)
            if not docstring:
                docstring_placeholder = ""
            else:
                docstring_placeholder = "\n###DCSTR###\n"
            scope = [arg.arg for arg in fun.args.args]
            for node in ast.walk(fun):
                # collecting scope
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    scope.append(node.id)
            scope = set(scope)
            first = True
            if len(scope) >= 3:
                for node in ast.walk(fun):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in scope:
                        id_ = node.id
                        a, b = node.first_token.startpos, node.last_token.endpos
                        node_code = precomment + code[a_fun: body_start] + docstring_placeholder + \
                        code[body_start: a] + special_word + code[b: b_fun] + postcomment
                        tokens, idx, remember_idxs, comments = get_tokens(node_code, special_word, id_, \
                                                                         docstring=docstring)
                        assert tokens[idx] == special_word
                        assert np.all([tokens[remem_idx]==id_ for remem_idx in remember_idxs])
                        
                        node.id = id_
                        if first:
                            tokens[idx] = id_
                            data.append([filenm, code[a_fun: b_fun], tokens, comments, list(scope), []])
                            first = False
                        
                        buggy_id = list(scope.difference({id_})) \
                                   [np.random.randint(low=0, high=len(scope) - 1)]
                        data[-1][-1].append([id_, buggy_id, idx, remember_idxs]) # correct, buggy
            if not first and len(data[-1][-1]) <= 1:
                del data[-1]

files = []
for r, d, f in os.walk(folder):
    for file in f:
        if file.endswith(".py"):
            files.append(os.path.join(r, file))
                
print("Processing", len(files), "files")

os.makedirs(new_folder, exist_ok=True)
os.makedirs(err_folder, exist_ok=True)
logs = [open(err_folder + "/errlog%d.txt" % proc, "w") for proc \
        in range(num_proc)]
outs = [open(new_folder + "/out%d.json" % proc, "w") for proc \
        in range(num_proc)]
portion = len(files) // (num_proc)
file_portions = [files[portion * proc: portion * (proc + 1)] \
                 for proc in range(num_proc)]
file_portions[-1] += files[portion * num_proc:]

def parallel_fun(files, log, out):
    processed = 0
    data = []
    for file in files:
        try:
            process_file(file, data)
        except Exception as e:
            log.write(traceback.format_exc() + "\n----------------\n")
        processed += 1
    out.write(json.dumps(data))

res = p_map(parallel_fun, file_portions, logs, outs, num_cpus=num_proc)

data = []
for proc in range(num_proc):
    with open(new_folder + "/out%d.json" % proc, "r") as outf:
        data_proc = json.loads(outf.read())
        data += data_proc

with open(new_folder + "/out_all.json", "w") as out:
    out.write(json.dumps(data))
for fn, source, tokens, comments, scope, pairs in data:
    assert len(pairs) > 1
    for pair in pairs:
        iden1, iden2, loc_idx, rep_idxs = pair
        assert len(scope) >= 3
        assert iden1 in scope
        assert iden2 in scope
        assert tokens[loc_idx] == iden1
        for rep_idx in rep_idxs:
            assert tokens[rep_idx] == iden1
        assert len(rep_idxs) >= 1
