# Literature Surveys

This directory receives structured literature-search outputs. A survey should state:

- research question and exclusion boundary;
- databases/search endpoints, query strings, and search date;
- candidate source list and screening reasons;
- source versions, URLs/DOIs, licenses where relevant, and retrieval hashes;
- extracted claims with page/section anchors;
- agreements, contradictions, missing evidence, and one next retrieval action.

Claude's specialized research workflow can later be integrated through this output contract. Its internal stages do not need to match `researchctl`; only the exported source ledger, screening record, extraction table, synthesis, and unresolved questions need stable formats.

Search results and memory-based recommendations begin as `lead`. They become `third_party`, `author`, or `primary` only after direct source verification.
