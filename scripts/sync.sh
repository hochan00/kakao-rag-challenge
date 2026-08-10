#!/usr/bin/env zsh
# 현재 브랜치에 원격으로 push된 변경만 안전하게 반영합니다.
# 자동 rebase, reset, stash, 강제 덮어쓰기는 수행하지 않습니다.

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print "사용법: ./scripts/sync.sh"
  print "현재 브랜치의 upstream 변경을 fetch한 뒤 fast-forward 방식으로만 반영합니다."
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  print -u2 "오류: Git 저장소 안에서 실행하세요."
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  print -u2 "중단: 커밋하지 않았거나 스테이징하지 않은 변경이 있습니다."
  print -u2 "먼저 git status로 변경을 확인하고 커밋 또는 별도 보관한 뒤 다시 실행하세요."
  exit 1
fi

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  print -u2 "오류: detached HEAD 상태에서는 동기화할 수 없습니다. 브랜치를 checkout하세요."
  exit 1
fi

if ! upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
  print -u2 "오류: '$branch' 브랜치에 upstream이 없습니다."
  print -u2 "예: git push -u origin $branch"
  exit 1
fi

print "동기화 대상: $branch <- $upstream"
print "원격 변경을 확인합니다..."
git fetch --prune origin

read -r behind ahead <<< "$(git rev-list --left-right --count "HEAD...$upstream")"
if (( behind == 0 && ahead == 0 )); then
  print "이미 최신 상태입니다."
  exit 0
fi

print "원격에 $behind개, 로컬에 $ahead개 커밋 차이가 있습니다."
print "fast-forward 병합을 시도합니다..."
if ! git pull --ff-only; then
  print -u2 "중단: fast-forward로 반영할 수 없습니다."
  print -u2 "현재 브랜치와 원격 브랜치가 갈라졌습니다. 변경을 검토한 뒤 직접 merge 또는 rebase를 선택하세요."
  exit 1
fi

print "동기화 완료"
git status -sb
