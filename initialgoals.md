# figregistry v0.3 – **Specification Draft** (Phase 1)

*Status: **DRAFT** – for review & approval by maintainers*

---

## 1  Goals of v0.3

1. **Config‑driven styling**: allow per‑project YAML to override default line/marker/colour styles, figure sizes, and folder aliases.
2. **Condition‑aware styles**: map experimental condition keys → style dictionaries once; reuse across all plots.
3. **Backward compatibility**: existing projects that only call `save_figure`/`saveFigure` without a YAML file must keep working (defaults suffice).
4. **Symmetry**: identical semantics in Python (Matplotlib) and MATLAB.

---

## 2  Top‑level YAML Structure

```yaml
figregistry_version: ">=0.3"   # semantic constraint; loader warns if unmet
style:
  backend: agg                 # matplotlib only; ignored by MATLAB
  rcparams:                    # Generic rcParams or MATLAB graphics defaults
    font.family: sans-serif
    font.size: 9
  palette:                     # Named colours usable anywhere
    primary: "#00628B"
    secondary: "#E36F1E"
layout:
  width_cm: 8.9                # float ≥ 0; golden‑ratio height if null
  height_cm: null
  dpi: 300                     # int ≥ 72
paths:
  root: figures                # default root folder
  purposes:                    # alias → canonical folder
    expl: exploratory
    pres: presentation
    pub: publication
naming:
  timestamp_fmt: "%Y-%m-%dT%H-%M-%S"  # strftime
  slug_pattern: "{ts}_{name}"          # {ts} & {name} placeholders mandatory
metadata:
  require_tags: false
condition_styles:              # free‑form keys, validated later
  control:
    color: "#A0A0A0"
    marker: "^"
    linestyle: "--"
    label: "Sham control"
  condition_a:
    color: "#00628B"
    marker: "o"
    linestyle: "-"
    label: "Optogenetic ON"
  mutant_px:
    color: "#E36F1E"
    marker: "s"
    linestyle: "-"
    label: "Px / Px mutant"
```

### 2.1  Required vs Optional

| Field                   | Required?              | Default             | Notes                                                                                                                                        |
| ----------------------- | ---------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `figregistry_version`   | **Yes**                | —                   | String with \[PEP 440] constraint.                                                                                                           |
| `style.backend`         | No                     | `agg`               | Ignored by MATLAB.                                                                                                                           |
| `style.rcparams`        | No                     | `{}`                | Allows any valid Matplotlib rcParam or MATLAB graphics root property.                                                                        |
| `style.palette`         | No                     | `{}`                | Named colours become accessible via `get_style(palette.primary)` if desired.                                                                 |
| `layout.width_cm`       | No                     | `8.9`               | If omitted, default width = 8.9 cm (1‑col journal).                                                                                          |
| `layout.height_cm`      | No                     | `null`              | `null` triggers golden ratio.                                                                                                                |
| `layout.dpi`            | No                     | `300`               | Used by `save_figure`; MATLAB calls `print -r300`.                                                                                           |
| `paths.root`            | No                     | `figures`           | Root relative to project CWD.                                                                                                                |
| `paths.purposes`        | No                     | see defaults        | Aliases can shorten calls (e.g. `pub` → `publication`).                                                                                      |
| `naming.timestamp_fmt`  | No                     | ISO 8601 w/ hyphens | Must contain `%Y` etc.                                                                                                                       |
| `naming.slug_pattern`   | No                     | `{ts}_{name}`       | Must include `{ts}` and `{name}` tokens.                                                                                                     |
| `metadata.require_tags` | No                     | `true`              | If `false` YAML side‑car may have empty `tags:` list.                                                                                        |
| `condition_styles`      | **Yes** (can be empty) | `{}`                | Keys validated as identifiers (letters, digits, underscores). Each value may include `color`, `marker`, `linestyle`, `label` (all optional). |

---

## 3  Pydantic Reference Model (Python)

```python
from pydantic import BaseModel, Field, validator

class Style(BaseModel):
    backend: str = "agg"
    rcparams: dict[str, str | int | float | bool] = Field(default_factory=dict)
    palette: dict[str, str] = Field(default_factory=dict)

class Layout(BaseModel):
    width_cm: float = 8.9
    height_cm: float | None = None
    dpi: int = 300

class Paths(BaseModel):
    root: str = "figures"
    purposes: dict[str, str] = Field(default_factory=lambda: {
        "exploratory": "exploratory",
        "presentation": "presentation",
        "publication": "publication",
    })

class Naming(BaseModel):
    timestamp_fmt: str = "%Y-%m-%dT%H-%M-%S"
    slug_pattern: str = "{ts}_{name}"

    @validator("slug_pattern")
    def must_contain_tokens(cls, v):
        if "{ts}" not in v or "{name}" not in v:
            raise ValueError("slug_pattern must include {ts} and {name} tokens")
        return v

class MetaOpts(BaseModel):
    require_tags: bool = True

class ConditionStyle(BaseModel):
    color: str | None = None
    marker: str | None = None
    linestyle: str | None = None
    label: str | None = None

class ConfigModel(BaseModel):
    figregistry_version: str = Field(..., regex=r"^[0-9><=~!.,*+^$\s]+$")
    style: Style = Style()
    layout: Layout = Layout()
    paths: Paths = Paths()
    naming: Naming = Naming()
    metadata: MetaOpts = MetaOpts()
    condition_styles: dict[str, ConditionStyle] = Field(default_factory=dict)
```

---

## 4  MATLAB Struct Reference

```matlab
cfg = struct( ...
    "figregistry_version", ">=0.3", ...
    "style", struct( ...
        "backend", "agg", ...   % ignored
        "rcparams", struct( "font.family", "sans-serif", "font.size", 9 ), ...
        "palette", struct( "primary", "#00628B" ) ...
    ), ...
    "layout", struct( "width_cm", 8.9, "height_cm", [], "dpi", 300 ), ...
    "paths", struct( "root", "figures", "purposes", struct("pub", "publication") ), ...
    "naming", struct( "timestamp_fmt", "%Y-%m-%dT%H-%M-%S", "slug_pattern", "{ts}_{name}" ), ...
    "metadata", struct( "require_tags", true ), ...
    "condition_styles", struct( "control", struct("color", "#A0A0A0", "marker", "^" ) ) ...
);
```

Validation implemented via helper `figregistry.validateConfig(cfg)`; errors aggregate into a cell‑array of strings.

---

## 5  Open Questions

1. **Palette fallback hierarchy** – should unnamed conditions pull from `style.palette` *or* Matplotlib cycle?
   *Current proposal*: yes, fallback to palette order.
2. **MATLAB rcparam parallels** – which keys do we honour? Propose subset: `font.size`, `LineWidth`, `AxesColorOrder`.
3. **`label` field** – required?
   *Proposal*: optional; used by legend helper.
4. **Golden‑ratio height** – lock to ϕ (1.618) or allow custom ratio?
   *Proposal*: keep simple: if `height_cm == null` use `width_cm / 1.618`.

Please comment via inline GitHub review or Slack thread **#figregistry‑dev**.

---

## 6  Approval Checklist (Task 1.2)

* [ ] Field names & defaults finalised
* [ ] Pydantic model compiles, unit tests pass
* [ ] MATLAB struct validation draft passes sample config
* [ ] Example YAML committed to `figregistry_spec/examples/` directory
* [ ] Maintainers 👍  (Sam B., Avery L., Kofi M.)

When all items are checked, Phase 1 is **done** and implementation tasks 2.x & 3.x may proceed.
