# RenamePDF

Identify and organize scientific-paper PDFs using DOI, title, and bibliographic
evidence. RenamePDF also detects duplicates, reports publication updates and
commentary, and finds replacements for eligible papers.

Dry-run is the default. Review the CSV report before changing a collection.

| Command | Purpose | Change files |
| --- | --- | --- |
| `RenamePDF.py` | Identify, rename, and organize papers | `--commit` |
| `DedupePDF.py` | Detect duplicates and preserve extra copies | `--commit`; normalization alone: `--preprocess` |
| `AuditPDF.py` | Audit identity, publication status, and PDF condition | Read-only |
| `ReplacePDF.py` | Find and validate audit-selected replacements | `--commit` |

Collection scans are recursive. The former `ReplacePublished.py` workflow is
replaced by AuditPDF and ReplacePDF. Use each command's `--help` for all options.

## Requirements and setup

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

From the repository directory:

```bash
uv sync --locked
```

Runtime dependencies are HTTPX, PyMuPDF, and truststore. PyMuPDF is pinned to
`1.28.2`; `pyproject.toml` and `uv.lock` define the supported environment.
Poppler, qpdf, and OCR software are not required by these commands.

For a checkout in a synced folder, set `UV_PROJECT_ENVIRONMENT` to an absolute
path outside that folder before running uv. Keep credentials in environment
variables; never put real passwords or API keys in commands saved to source.

## Organize papers

Preview proposed names and placement:

```bash
uv run python -OO RenamePDF.py /path/to/papers --email you@example.org
```

Review `0_results.csv`, including identity, confidence, PDF status, audit fields,
and target paths. Then authorize the changes:

```bash
uv run python -OO RenamePDF.py /path/to/papers --email you@example.org --commit
```

Useful options:

- `--newer-than-results`: select PDFs newer than the existing report.
- `--retry-failed`: select PDFs under `Failed/`.
- `--retry-audit-incomplete`: refresh unfinished publication checks in the
  existing report without reopening or moving PDFs.
- `--inspection-timeout SECONDS`: adjust the read-only inspection deadline;
  the default is 120 seconds. AuditPDF supports this option too.
- `--ncbi-api-key "$NCBI_API_KEY"`: supply an optional NCBI key.

Set `RENAMEPDF_PDF_PASSWORD` when one password should be tried for encrypted PDFs.

## Detect duplicates

```bash
uv run python -OO DedupePDF.py /path/to/papers
```

Review `0_dedupe_results.csv`, then repeat with `--commit` to preserve detected
copies under `Duplicated/`. Files are not deleted.

`--fast` inspects only groups sharing the first seven filename characters;
other rows are marked `skipped_fast_prefix`, not confirmed unique. Use the
default exhaustive mode for a complete scan. `--preprocess` separately permits
repair, decryption, and recognized-cover removal without duplicate placement.

## Audit and replace papers

Create a read-only audit:

```bash
uv run python -OO AuditPDF.py /path/to/papers --email you@example.org
```

Review `0_audit_results.csv`. It includes:

- Identity and document type, such as preprint, proceeding, or journal article.
- Retractions, corrections, expressions of concern, republications, and
  reinstatements from PubMed and Crossref.
- PubMed linked comments and Scite supporting, contradicting, and mentioning
  citation counts. PubPeer is not queried.
- PDF access, repair needs, readable-text limitations, and replacement actions.

Formal updates require human review. `peer_review_not_established` records
uncertainty; it does not mean that peer review did not occur. Missing provider
responses produce incomplete checks. An absence of reported signals does not
establish that no notice or criticism exists.

Preview replacements selected by the current audit report:

```bash
uv run python -OO ReplacePDF.py /path/to/papers --email you@example.org
```

Review `0_replace_results.csv`, then repeat with `--commit` if the proposed
replacements are correct. Eligible actions search for a later publication of a
preprint or proceeding, or a readable copy of the same accepted DOI for an
image-only PDF. The tool rechecks eligibility and source state, validates
existing targets and downloads, and preserves replaced sources in `Superseded/`.

## Preservation and privacy

- Ambiguous or conflicting evidence remains unresolved.
- Required repair, decryption, and recognized-cover removal precede identity
  acceptance. Signed PDFs that require rewriting are blocked.
- Source changes after inspection block mutation. Existing destinations are
  validated or preserved; files are never silently overwritten or deleted.
- Commit may losslessly compress PDFs larger than decimal 3 MB, installing
  output only when it is validated and smaller.
- Interrupted rewrites and replacements preserve recovery artifacts. Keep
  journals and surviving copies until the reported recovery state is resolved.
- Dry-run leaves source PDFs unchanged but may update reports and caches.
- Providers receive identifiers and search terms, never PDF bytes. Reports and
  caches can contain private paths and publication metadata; keep them private.

Run mutating commands one at a time and use one active computer for a synced
collection. Local locks do not coordinate separate computers. PyMuPDF worker
isolation is not an operating-system security sandbox.

## Reports and folders

| Path | Purpose |
| --- | --- |
| `0_results.csv` | Organization report |
| `0_dedupe_results.csv` | Duplicate report |
| `0_audit_results.csv` | Publication audit and replacement eligibility |
| `0_replace_results.csv` | Replacement resolution and installation report |
| `Failed/` | Classified unresolved PDFs |
| `Duplicated/` | Preserved duplicate copies |
| `Superseded/` | Preserved replaced sources |

Bibliographic responses use `.pdf_renamer_cache.sqlite`. PDF evidence uses a
local temporary working cache and a completed
`.renamepdf_identity_cache.sqlite` snapshot beside the collection. Cache reuse
requires matching file state and extractor version; reports and fresh checks
still govern acceptance and mutation.

Generated filenames use author names, venue, year, and keywords, with portable
characters and a 250-byte UTF-8 limit including `.pdf`.

## Development

The four CLI scripts adapt arguments to reusable workflow modules:

| Responsibility | Modules |
| --- | --- |
| Organization and read-only audits | `organization_run.py`, `audit_report.py` |
| Duplicates and replacements | `dedupe_run.py`, `replacement_run.py` |
| Evidence, normalization, worker processes | `pdf_evidence.py`, `pdf_rewrite.py`, `pdf_process.py` |
| Providers, identity, publication review | `metadata_providers.py`, `record_matching.py`, `publication_audit.py`, `publication_status.py` |
| Classification and filenames | `document_classification.py`, `filename_policy.py` |
| Placement, metadata, caches, reports, locks | `collection_mutation.py`, `file_metadata.py`, `pdf_identity_cache.py`, `report_io.py`, `file_lock.py` |

Use synthetic PDFs and mocked providers. Run the smallest affected test suite
without `-OO`, for example:

```bash
uv run python -m unittest tests/test_audit_replace.py -q -b
```

For broad changes or release preparation:

```bash
uv lock --check
uv run python -m unittest discover -s tests -q -b
git diff --check
```

Also check installed dependency compatibility and all four CLI `--help`
commands before release. Regenerate `requirements.txt` from the lockfile with
`uv export --locked --format requirements-txt --no-hashes -o requirements.txt`.
See the [pull-request checklist](.github/pull_request_template.md).

## Limitations and reporting

Provider coverage can be incomplete or incorrect. Image-only PDFs may need
external OCR when no usable identifier or title is embedded. Only recognized
inserted covers are removed; downloads larger than 512 MiB are rejected.
Metadata matching does not establish scientific quality or copyright status.

Report security issues privately through GitHub vulnerability reporting or a
private Security Advisory when available. Use a sanitized reproduction and
never attach a confidential PDF.

No reuse license has been selected. Confirm the approved license and repository
visibility before public release.
