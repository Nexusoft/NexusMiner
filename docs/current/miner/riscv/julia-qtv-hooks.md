# Julia QTV and C++ Hook Boundaries

This note captures the recommended hook shape for the `julia/NexusMinerQTV`
research package and any future C++ integration point.

## Julia package layout

The Julia side should remain a normal package with explicit source/test
boundaries:

```text
julia/NexusMinerQTV/
├── Project.toml
├── Manifest.toml
├── README.md
├── src/
│   ├── NexusMinerQTV.jl
│   ├── qtv.jl
│   ├── parity.jl
│   └── hooks.jl
└── test/
```

## Julia module responsibilities

The Julia layer is the right place for:

- deterministic fixture generators
- QTV math/vector kernels
- parity replay helpers
- small C-callable fixture/parity entry points
- explicit, opt-in RISC-V experiments behind dedicated modules

It is **not** the place for live miner networking, session ownership, auth flow,
or block-submission dependencies.

## Narrow C-callable surface

The exported library boundary should stay small and explicit:

```julia
Base.@ccallable function qtv_run_fixture(case_id::Cint)::Cint
Base.@ccallable function qtv_compare_parity(case_id::Cint)::Cint
```

Both hooks operate on deterministic fixtures only and return explicit status
codes.  That keeps Julia interop reproducible and safe to disable outside of
research or benchmark workflows.

## C++ side: one adapter boundary

Do not spread Julia interop across miner/runtime code.  Keep a single adapter
boundary instead:

```cpp
class QTVJuliaBridge
{
public:
    bool available() const;
    int run_fixture(int case_id);
    int compare_parity(int case_id);
};
```

This isolates libjulia-specific failure modes and keeps production runtime
protection simple.

The formal backend-selection and fallback contract now lives in:

- `docs/current/node/riscv/JuliaProgrammingLanguage/bridge-architecture.txt`
- `include/qtv/QTVBackendKind.hpp`
- `include/qtv/QTVCapabilities.hpp`

Those files define the public backend enum plus the deterministic selection
rules that preserve C++ as the production authority while keeping Julia opt-in
for fixture/parity research.

## C++ side: interface + backends

When you need a swap-engine abstraction, prefer a small interface plus concrete
backends:

```cpp
class IQTVEngine
{
public:
    virtual ~IQTVEngine() = default;
    virtual bool RunFixture(int caseId) = 0;
    virtual bool CompareParity(int caseId) = 0;
};
```

Recommended backends:

- `CppQTVEngine` for local scalar/SIMD parity or benchmark logic
- `JuliaQTVEngine` for the library bridge
- `NullQTVEngine` for production builds that intentionally disable Julia hooks

## Swap-engine model

Treat backend choice as an explicit mode selection:

- build flag
- test harness mode
- benchmark mode
- research mode

Do **not** route the default production miner path through Julia.  `Auto`
selection must prefer the C++ backend first and only fall back to Julia when
the full deterministic research surface is available and C++ is not.

## RISC-V hook framing

RISC-V hooks here mean Julia can:

- benchmark vector-width candidates
- generate deterministic tables/constants
- replay parity fixtures across architectures
- explore math kernels that may inform later C++ implementations

They do **not** mean the live miner depends on Julia at runtime.
