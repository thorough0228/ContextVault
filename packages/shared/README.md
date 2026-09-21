# packages/shared

Shared types and constants across the API, worker, and web app. The
goal is to keep names of routes, status codes, and versioned protocol
constants in lock-step across all three runtimes.

Layout:

```
packages/shared/
├── python/contextvault_shared/        # Python import (api + worker)
│   ├── __init__.py
│   └── constants.py
└── ts/                            # TypeScript import (web)
    └── index.ts
```

Both sides mirror the same constants — Python and TS — so each side can
import the language-native version. Editing one *must* be mirrored in
the other. There is no codegen in Phase 1; keep them in sync manually.