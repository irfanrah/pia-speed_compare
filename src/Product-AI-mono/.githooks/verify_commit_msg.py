#!/usr/bin/python3

import re
import sys

type_list = ["feat", "fix", "refactor", "style", "docs", "test", "chore", "HOTFIX", "revert"]
type_regex = r"^(feat|fix|refactor|style|" r"docs|test|chore|HOTFIX|revert)(\(.+\))?:\s(.{3,})"


class bcolors:
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def is_merge_commit(lines):
    # 예: "Merge branch ..." 같은 메시지 탐지
    for line in lines:
        if line.strip().upper().startswith("MERGE"):
            return True
    return False


def verify_commit_message():
    with open(sys.argv[1]) as commit:
        lines = commit.readlines()
        # lines = ["\n","feat: Asdasd", "\n", "\n"]
        # Remove comments
        lines = [line for line in lines if not line.startswith("#")]
        # If the last line is whitespace, remove it
        lines = [line for line in lines if line.strip() != ""]

        # Empty commit message
        if len(lines) == 0:
            sys.stderr.write(f"\n{bcolors.FAIL} Commit failed:\n")
            sys.stderr.write(f"{bcolors.ENDC}Empty commit message.\n")
            sys.exit(1)

        first_line = lines[0].strip()

        # Merge Commit is allowed.
        if is_merge_commit([first_line]):
            sys.exit(0)

        # Subject line should be less than 50 characters.
        if len(lines[0]) > 50:
            sys.stderr.write(f"\n{bcolors.FAIL}Commit failed:\n")
            sys.stderr.write(f"{bcolors.ENDC}Subject line should be less than 50 characters.\n")
            sys.exit(1)

        # Subject line should follow the rule.
        if re.match(type_regex, lines[0]) is None:
            sys.stderr.write(f"\n{bcolors.FAIL}Commit failed:\n")
            sys.stderr.write(
                f"{bcolors.ENDC}The commit message subject line does not follow the rule."
            )
            sys.stderr.write("\n<type>: <Subject> is required.\n")
            sys.exit(1)

        for line in lines[2:]:
            # Every single description should be less than 72 characters.
            if len(line) > 72:
                sys.stderr.write("\nEvery single description should be less than 72 characters.\n")
                sys.exit(1)
            # Description starts with "-".
            if not line.startswith("-"):
                sys.stderr.write(line)
                sys.stderr.write(f"\n{bcolors.FAIL} Commit failed:\n")
                sys.stderr.write(f"{bcolors.ENDC}Description should start with a dash '-'.\n")
                sys.exit(1)

        sys.exit(0)


if __name__ == "__main__":
    verify_commit_message()
