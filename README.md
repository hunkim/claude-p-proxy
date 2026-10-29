# claude-p-proxy

> **개인용입니다.** 내 Claude Pro/Max 구독으로 내 컴퓨터에서 나 혼자 쓰는 용도로 만들었습니다.

`claude -p`(Claude Code CLI)를 OpenAI / Anthropic API 호환으로 감싼 로컬 서버.
SDK에서 `base_url`만 바꾸면 그대로 붙습니다. 텍스트, 스트리밍, 이미지 입력, function calling 지원.

## 설치·실행

```bash
git clone https://github.com/hunkim/claude-p-proxy && cd claude-p-proxy
npm install        # Claude Code CLI도 함께 설치됨
npx claude         # 처음 한 번: Pro/Max 계정으로 로그인 후 종료
npm start          # http://127.0.0.1:8787
```

이미 `claude`를 전역 설치해 쓰고 있다면 `npm install` 없이 `node server.mjs`만 해도 됩니다.

## 사용

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="local")
print(c.chat.completions.create(model="sonnet",
      messages=[{"role": "user", "content": "안녕"}]).choices[0].message.content)
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:8787", api_key="local")
print(c.messages.create(model="opus", max_tokens=1024,
      messages=[{"role": "user", "content": "안녕"}]).content[0].text)
```

Open WebUI, LibreChat 같은 채팅 UI에는 OpenAI 호환 주소로 `http://127.0.0.1:8787/v1`을 넣으면 됩니다.

## 엔드포인트

- `POST /v1/chat/completions` — OpenAI 호환
- `POST /v1/messages` — Anthropic 호환
- `GET /v1/models`, `GET /health`

모델 이름에 `opus`/`sonnet`/`haiku`/`fable`이 들어가면 그 모델로, `claude-sonnet-5-5`처럼 버전까지 적으면 그대로,
`gpt-4o` 같은 나머지는 `DEFAULT_MODEL`로 갑니다. `reasoning_effort`(low~max)는 `--effort`로 넘어갑니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8787` | |
| `DEFAULT_MODEL` | `sonnet` | |
| `MAX_CONCURRENT` | `2` | 동시에 띄울 `claude` 프로세스 수 |
| `LOCAL_API_KEY` | (없음) | 설정하면 `Authorization: Bearer` 또는 `x-api-key` 확인 |
| `TIMEOUT_MS` | `300000` | 요청당 제한 시간 |
| `DEFAULT_SYSTEM` | 간단한 지시문 | system 메시지가 없을 때 |
| `MAX_IMAGE_BYTES` | 10MB | URL 이미지 최대 크기 |
| `CLAUDE_BIN` | 자동 | CLI 경로 (기본: `node_modules`의 CLI → PATH의 `claude`) |

## 동작 방식

- 요청마다 `claude -p`를 띄웁니다. Claude Code 내장 도구·MCP·세션 저장은 끄고 빈 임시 폴더에서 실행합니다.
- 대화 기록은 텍스트 하나로 합치고, 이미지는 base64 이미지 블록으로 붙여 `--input-format stream-json`으로 넘깁니다. URL 이미지는 서버가 내려받습니다.
- Function calling: 도구 정의를 시스템 프롬프트에 넣고 `<tool_calls>[...]</tool_calls>` JSON 블록으로 답하게 한 뒤, 서버가 꺼내 `tool_calls` / `tool_use` 형식으로 돌려줍니다. 인자가 스키마에 맞는다는 보장은 없으니 중요한 곳에서는 검증하세요.
- `ANTHROPIC_API_KEY`는 자식 프로세스에서 지워 구독 한도만 씁니다.

## 한계

- 첫 토큰까지 수 초 (프로세스 시작 비용)
- temperature·max_tokens·stop 등은 무시, PDF·오디오 입력 미지원
- 사용량은 Claude 앱·Claude Code와 같은 구독 한도에서 빠집니다
