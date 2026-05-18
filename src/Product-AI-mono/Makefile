init:
	@echo "🔧 pre-commit 설치 중..."
	pip install pre-commit

	@echo "⚙️  Git 설정 적용 중..."
	git config commit.template .githooks/.gitmessage

	@echo "🔧 pre-commit 훅 설치 중..."
	pre-commit install
	pre-commit install-hooks

	@echo "commit convention 설정 중..."
	cp .githooks/commit-msg .git/hooks/commit-msg
	chmod +x .git/hooks/commit-msg
	@echo "✅ Git 커밋 템플릿 및 pre-commit hook이 성공적으로 설정되었습니다."
