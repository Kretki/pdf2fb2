# AGENTS.md

Rules for any agent (or human) creating or modifying code in this repository.
Every rule marked **MUST** is mandatory: code that violates it is not done.

Project layout, for orientation:

- `src/` — TypeScript front end (Vite, WebGL globe), entry point `src/application.ts`:
  `geodesy/` WGS84 math, `land/` loading and meshing, `plates/` plate reconstruction,
  `rendering/` WebGL, `camera/` orbit camera and input, `ui/` DOM panels, `app/` controllers wiring them together.
- `server/` — Python FastAPI back end: `application.py` (wiring) plus packages `configuration/` (lod.json settings),
  `contracts/` (protocols), `http_api/` (routes), `land_data/` (GeoPackage queries), `plate_model/` (rotations,
  plate layers, topologies), `paleomagnetism/` (PINT sites, TPW data) and `shared_utils/`.
- `tools/` — Python data-preparation scripts (`plate_tag.py` + `plate_tagging/`, `gpml_to_geojson.py`).
- `public/data/` — shell scripts and plate-model data.

Follow this instructions with minimum and only-necessary output in the console for coder. Coder should know only aggregated info about changes at the end and about the descions you struck into that needs coder infiltration and errors that needs to resolve with coder.

---

## 1. Naming: 8-character minimum

- Every identifier you introduce — variables, constants, parameters, function and method names, class names, module/file names — **MUST** be at least **8 characters** long.
- Names must be descriptive, not padded: `latitude`, `vertexCount`, `rotation_pole`, not `latxxxxx` or `lat_____`.
- Loop counters are not exempt: use `rowIndex`, `pointIdx` (8 chars) instead of `i`, `j`, `k`.
- Exceptions (only these):
  - names fixed by an external API, framework or protocol (e.g. FastAPI's `app`, `self`, `cls`, `__init__`, DOM `id`, GLSL built-ins);
  - the throwaway `_`.
- Follow the language's case convention: `camelCase` / `PascalCase` in TypeScript, `snake_case` / `PascalCase` in Python, `UPPER_SNAKE_CASE` for constants in both.

## 2. Docstrings for every function

Every function, method and class **MUST** have a docstring that explains:

1. **Usage** — what it does and when to call it.
2. **Input variables** — each parameter, its type, units and valid range where relevant.
3. **Output data** — the return type and meaning, and anything it raises/throws or mutates.

Python (Google style):

```python
def rotate_coordinates(latitude_deg: float, longitude_deg: float, rotation_pole: EulerPole) -> tuple[float, float]:
    """Rotate a surface point around a finite Euler pole.

    Usage:
        Call when reconstructing a present-day point to a past age.

    Args:
        latitude_deg: Latitude of the point in degrees, -90..90.
        longitude_deg: Longitude of the point in degrees, -180..180.
        rotation_pole: Euler pole (lat, lon, angle in degrees) to apply.

    Returns:
        Tuple (latitude_deg, longitude_deg) of the rotated point in degrees.

    Raises:
        ValueError: If the input coordinates are out of range.
    """
```

TypeScript (TSDoc):

```ts
/**
 * Build a triangulated mesh for one land polygon at the given LOD.
 *
 * Usage: called by the mesh worker for every polygon in a loaded tile.
 *
 * @param polygonRings - Outer ring followed by hole rings, as [lon, lat] degree pairs.
 * @param detailLevel - LOD index from `lod.config.ts` (0 = coarsest).
 * @returns Vertex positions (Float32Array, xyz on the unit sphere) and triangle indices (Uint32Array).
 * @throws Error if the outer ring has fewer than 3 points.
 */
```

Shell scripts **MUST** start with a header comment covering usage, arguments and outputs.

## 3. Logical blocks and file distribution

- A function does **one thing**. If it needs comments like "step 1 / step 2", split each step into its own function.
- Keep functions short (aim for ≤ 40 lines) and nesting shallow (≤ 3 levels).
- Group related functions into a module by responsibility; one module = one concern. Examples:
  - parsing / I/O (`gpkg_store.py`, `geometry_decoder.py`, `landApiClient.ts`)
  - pure computation (`rotation_tree.py`, `meshBuilder.ts`, `rectClipping.ts`)
  - configuration (`lod_config.py`, `lod.config.ts`)
  - transport / API layer (`api_routes.py`)
  - rendering (`globeRenderer.ts`, `selectionOverlay.ts`)
- When a file mixes concerns or grows past ~300 lines, split it into new files by concern.
- No logic in API route handlers or entry points (`application.py`, `api_routes.py`, `application.ts`) beyond wiring: they delegate to service modules.
- No circular imports between modules.

## 4. SOLID principles

All code **MUST** follow SOLID:

- **S — Single Responsibility:** each class, function and module has one reason to change (see section 3).
- **O — Open/Closed:** extend behaviour through new implementations (e.g. a new data source or LOD strategy) rather than editing existing branches; avoid growing `if/else` or `switch` chains on type.
- **L — Liskov Substitution:** subclasses and interface implementations must be usable wherever their base type is expected, without changing pre/post-conditions.
- **I — Interface Segregation:** prefer small, focused interfaces / `Protocol`s over large ones; callers depend only on the methods they use.
- **D — Dependency Inversion:** high-level modules depend on abstractions (TypeScript interfaces, Python `Protocol`/ABC), and concrete dependencies (file readers, HTTP clients, config) are injected rather than created inside the code that uses them.

## 5. Security linting (SEC-linting)

Code **MUST** pass security linting with zero unresolved findings before it is considered complete.

- **Python:** `bandit` (security linter) and `ruff` with the `S` (flake8-bandit) rule set enabled.
- **TypeScript:** ESLint with `eslint-plugin-security` (`eslint.config.js`; it also enforces the 8-character `id-length` rule and bans `innerHTML`/`eval`).
- **Shell:** `shellcheck`.
- Secure-coding rules, regardless of tool output:
  - never use `eval`, `exec`, `new Function`, `pickle` on untrusted data, or `shell=True` with interpolated input;
  - validate and bound every external input (query parameters, file paths, uploaded or downloaded data); reject path traversal (`..`) in any user-supplied path;
  - no hard-coded secrets, tokens or credentials; read them from environment variables;
  - no `innerHTML` with untrusted content;
  - parameterise all SQL (including GeoPackage/SQLite queries); never build queries by string concatenation.
- A finding may be suppressed only with an inline comment that names the rule and states why it is a false positive (e.g. `# nosec B608 - table name comes from fixed allowlist`).

## 6. Mandatory security and dependency checks

Before finishing any change, run the checks below and fix every finding of severity **medium or higher**. Do not report a task as done while any check fails; if a tool is not installed, say so explicitly instead of skipping silently.

| Tool | Purpose | Command |
|------|---------|---------|
| `pip-audit` | Known vulnerabilities in Python dependencies | `uv run pip-audit` and `uv run pip-audit -r server/requirements.txt` |
| `npm audit` | Known vulnerabilities in JS dependencies | `npm audit --audit-level=moderate` |
| `trivy` | Vulnerabilities, secrets and misconfigurations across the repo | `trivy fs --scanners vuln,secret,misconfig --severity MEDIUM,HIGH,CRITICAL --exit-code 1 --skip-dirs node_modules,.venv,dist .` |
| `semgrep` | Static analysis for security and correctness patterns | `uv run semgrep scan --config p/default --config p/python --config p/typescript --config p/secrets --exclude node_modules --exclude .venv --exclude dist --error` |
| `bandit` | Python security linting | `uv run bandit -r server tools -ll` |
| `ruff` | Python linting incl. security rules | `uv run ruff check server tools` (rules and line length in `pyproject.toml`) |
| `eslint` | TypeScript linting incl. security plugin | `npm run lint` |
| `shellcheck` | Shell script linting | `shellcheck public/data/*.sh` |
| `tsc` | Type checking | `npx tsc --noEmit` |

Additional dependency rules:

- Pin new dependencies to an explicit version or bounded range; do not add a dependency without checking it with `pip-audit` / `npm audit`.
- Prefer the standard library or existing dependencies over adding new ones.

## 7. Definition of done

A change is complete only when all of the following hold:

- [ ] All new identifiers are ≥ 8 characters (section 1).
- [ ] Every function, method and class has a docstring with usage, inputs and outputs (section 2).
- [ ] Functions are split into logical blocks and files by concern (section 3).
- [ ] The design follows SOLID (section 4).
- [ ] SEC-linting passes (section 5).
- [ ] `pip-audit`, `trivy`, `semgrep` and the other checks in section 6 pass, and their results are reported.
- [ ] Existing tests still pass (`npm run test:land`, `npm run build`).
