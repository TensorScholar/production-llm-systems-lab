# Real vLLM fixtures (T13 output — DATA REQUIRED)

Captured real fixtures land here after the T13 GPU compatibility check:

```text
tests/fixtures/vllm/<version>/
├── startup.log
├── metrics_idle.txt
├── metrics_busy.txt
└── version.json
```

Do not invent version-specific fixtures: parser tests run against the mock
server's output until T13 captures the real ones.
