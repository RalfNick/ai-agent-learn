# Memory Recall Review

Recall applies namespace isolation before relevance. Only active, unexpired records may be returned.

| Case | Decision | Recalled IDs |
| --- | --- | --- |
| `cross-tenant-recall` | `reject` | `none` |

The cross-tenant fixture returning `none` is the intended result, even when another tenant has a highly relevant preference.
