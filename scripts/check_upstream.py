"""F03 source provenance acceptance; standard library only, no upstream imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "compact_gab", "board_compression", "shared_template_bank", "input_projection",
    "absolute_pe", "post_layernorm", "dropout", "final_normalization", "initialization",
    "pair_policy", "promotion_adjustment", "vocabulary_ordering", "wdl_head",
    "make_rpe_map", "rpe_logits", "rpe_value", "qkv_contractions", "history_padding",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def get(url):
    request = Request(url, headers={"User-Agent": "RL-Chess-F03-source-check"})
    with urlopen(request, timeout=30) as response:
        require(response.status == 200, f"HTTP failure: {url}")
        return response.read()


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args])


def symbols(body, prefix=""):
    found = {}
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + node.name
            found[name] = (node.lineno, node.end_lineno)
            if isinstance(node, ast.ClassDef):
                found.update(symbols(node.body, name + "."))
    return found


def check(checkouts):
    lock = json.loads((ROOT / "docs/upstream-lock.json").read_text())
    require(lock["schema_version"] == 1, "unsupported lock schema")
    repos = lock["repositories"]
    require({k: r["identity"] for k, r in repos.items()} == {
        "maia3": "CSSLab/maia3", "lczero": "daniel-monroe/lczero-training",
    }, "repository identities changed")
    ids = [op["id"] for op in lock["operations"]]
    require(set(ids) == REQUIRED and len(ids) == len(REQUIRED), "missing/duplicate planned operation")
    data, file_checks, commit_checks = {}, [], {}
    for key, repo in repos.items():
        sha, identity = repo["commit"], repo["identity"]
        require(re.fullmatch(r"[0-9a-f]{40}", sha), f"not a full commit SHA: {key}")
        require(repo["git_url"] == f"https://github.com/{identity}.git", "git identity URL mismatch")
        require(repo["commit_url"] == f"https://github.com/{identity}/commit/{sha}", "moving commit URL")
        if checkouts:
            checkout = checkouts[key]
            require(git(checkout, "config", "--get", "remote.origin.url").decode().strip() == repo["git_url"], "checkout origin mismatch")
            resolved = git(checkout, "rev-parse", "--verify", sha + "^{commit}").decode().strip()
            tree = git(checkout, "ls-tree", "-r", "--name-only", sha).decode().splitlines()
        else:
            commit = json.loads(get(f"https://api.github.com/repos/{identity}/git/commits/{sha}"))
            resolved = commit["sha"]
            tree_result = json.loads(get(f"https://api.github.com/repos/{identity}/git/trees/{commit['tree']['sha']}?recursive=1"))
            require(not tree_result.get("truncated"), "incomplete repository tree")
            tree = [entry["path"] for entry in tree_result["tree"]]
        require(resolved == sha, f"commit object mismatch: {key}")
        license_paths = [p for p in tree if Path(p).name.upper().split('.')[0] in {"LICENSE", "COPYING", "NOTICE"}]
        require(license_paths == (["LICENSE"] if key == "maia3" else []), f"license inventory mismatch: {key}")
        commit_checks[key] = {"identity": identity, "commit": resolved, "license_paths": license_paths}
        for path, record in repo["files"].items():
            require(path in tree, f"path missing from pinned tree: {path}")
            require(record["blob_url"] == f"https://github.com/{identity}/blob/{sha}/{path}", "moving blob URL")
            require(record["raw_url"] == f"https://raw.githubusercontent.com/{identity}/{sha}/{path}", "moving raw URL")
            content = git(checkouts[key], "show", f"{sha}:{path}") if checkouts else get(record["raw_url"])
            digest = hashlib.sha256(content).hexdigest()
            require(digest == record["sha256"], f"checksum mismatch: {identity}/{path}")
            data[key, path] = content
            file_checks.append({"repository": key, "path": path, "sha256": digest})
    reference_count = 0
    for op in lock["operations"]:
        require(op["references"], f"operation has no reference: {op['id']}")
        for ref in op["references"]:
            key, path = ref["repository"], ref["path"]
            source_symbols = symbols(ast.parse(data[key, path]).body)
            require(ref["symbol"] in source_symbols, f"missing source symbol: {ref['symbol']}")
            start, end = ref["lines"]
            lo, hi = source_symbols[ref["symbol"]]
            require(lo <= start <= end <= hi, f"range outside symbol: {ref['symbol']}")
            expected = repos[key]["files"][path]["blob_url"] + f"#L{start}-L{end}"
            require(ref["url"] == expected, f"reference not immutable: {op['id']}")
            reference_count += 1
    for key, repo in repos.items():
        license_record = repo["license"]
        expected = data[key, license_record.get("notice_source", license_record["path"])]
        if "notice_lines" in license_record:
            start, end = license_record["notice_lines"]
            expected = b"".join(expected.splitlines(keepends=True)[start - 1:end])
        require((ROOT / license_record["notice"]).read_bytes() == expected, f"retained notice differs: {key}")
    return {"status": "passed", "mode": "git_objects" if checkouts else "https",
            "repositories": commit_checks, "files_verified": file_checks,
            "operations_verified": ids, "references_verified": reference_count,
            "retained_notices_verified": 2, "numerical_port_validation": "not performed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maia3-checkout", type=Path)
    parser.add_argument("--lczero-checkout", type=Path)
    args = parser.parse_args()
    require(bool(args.maia3_checkout) == bool(args.lczero_checkout), "supply both checkout paths or neither")
    checkouts = {"maia3": args.maia3_checkout, "lczero": args.lczero_checkout} if args.maia3_checkout else None
    print(json.dumps(check(checkouts), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"F03 source acceptance FAILED: {error}", file=sys.stderr)
        sys.exit(1)
