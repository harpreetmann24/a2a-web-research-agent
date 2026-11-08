# A2A Web Research Agent

A small LangGraph agent served over the [A2A protocol](https://github.com/a2aproject) with `a2a-sdk` 1.3.0.
It's a teaching prototype for an agents-in-production class.

- Agent card: `GET /.well-known/agent-card.json` (public)
- JSON-RPC endpoint: `POST /` (requires `Authorization: Bearer <A2A_TOKEN>`)
- Model: any OpenRouter model (default `anthropic/claude-haiku-5.5`)
- The `search_web` tool is a **mock** with a fixed set of headlines. It is a demo, not a real search service.

## Run locally

```
pip install -r requirements.txt
export OPENROUTER_API_KEY=... A2A_TOKEN=...
python web_research_agent.py        # http://127.0.0.1:9001/
```

## Deploy on Render

Free Docker web service; `render.yaml` describes it. Set these environment variables in Render:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | an OpenRouter key with a small credit limit |
| `A2A_TOKEN` | a long random string that callers must send as a Bearer token |
| `PUBLIC_URL` | the service URL, e.g. `https://a2a-web-research-agent.onrender.com/` |

Render provides `PORT`; the app listens on it.
