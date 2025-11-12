#!/usr/bin/env python3
"""
project_cloner.py

Usage:
    python3 project_cloner.py repos.txt
"""

import re
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

def check_git():
    try:
        subprocess.run(["git", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        sys.exit("❌ Git is not installed or not available on PATH.")

def parse_repos(file_path):
    """Parse lines like 'Team 25: https://github.com/...'"""
    pattern = re.compile(r"Team\s*(\d+)\s*:\s*(\S+)")
    repos = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if match:
                team_num, url = match.groups()
                repos.append((int(team_num), url))
            else:
                print(f"⚠️ Skipping malformed line: {line}")
    return repos

def is_accessible(url):
    """Check repo access via git ls-remote."""
    try:
        subprocess.run(
            ["git", "ls-remote", "--exit-code", url],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20
        )
        return True
    except Exception:
        return False

def clone_repo(url, dest, shallow=False):
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [url, dest]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, "cloned successfully"
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode(errors="ignore").strip()
    except Exception as e:
        return False, str(e)

def process_repo(team_num, url, shallow=False):
    dest = f"team{team_num}_vidyavichar"
    print(f"\n🔍 Checking Team {team_num}: {url}")
    if not is_accessible(url):
        return team_num, False, "inaccessible or invalid URL"
    print(f"✅ Repo accessible. Cloning into {dest} ...")
    success, msg = clone_repo(url, dest, shallow)
    return team_num, success, msg

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 clone_repos_by_team.py repos.txt [--shallow] [--workers N]")
    
    file_path = Path(sys.argv[1])
    shallow = "--shallow" in sys.argv
    workers = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])

    if not file_path.exists():
        sys.exit(f"❌ File not found: {file_path}")

    check_git()
    repos = parse_repos(file_path)
    if not repos:
        sys.exit("⚠️ No valid 'Team N: URL' lines found.")

    print(f"Found {len(repos)} repos. Shallow={shallow}, Workers={workers}")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_repo, t, u, shallow) for t, u in repos]
        for fut in as_completed(futures):
            team_num, success, msg = fut.result()
            status = "✅ CLONED" if success else "❌ FAILED"
            print(f"[Team {team_num}] {status}: {msg}")
            results.append((team_num, success))

    print("\n📊 Summary:")
    total = len(results)
    cloned = sum(1 for _, ok in results if ok)
    failed = total - cloned
    print(f"  Total teams: {total}")
    print(f"  Successfully cloned: {cloned}")
    print(f"  Failed / invalid: {failed}")

if __name__ == "__main__":
    main()
