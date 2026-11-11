# RISC-V CPU feature detection at library load time.
# On non-RISC-V platforms all boolean fields are false and counts fall back to Sys.CPU_THREADS.

struct RiscvCapabilities
    is_riscv          :: Bool    # true if running on any RISC-V hart
    has_v_ext         :: Bool    # true if "V" vector extension available
    has_zba           :: Bool    # true if Zba bit-manipulation extension available
    has_zbb           :: Bool    # true if Zbb bit-manipulation extension available
    hart_count        :: Int     # number of harts (RISC-V cores) detected
    xlen              :: Int     # register width (32 or 64)
    cache_line_bytes  :: Int     # detected or assumed cache-line size in bytes
    detected_at       :: String  # ISO-8601 timestamp string of detection
end

function _parse_riscv_cpuinfo()
    has_v   = false
    has_zba = false
    has_zbb = false
    hart_count = 0

    try
        for line in eachline("/proc/cpuinfo")
            lc = lowercase(strip(line))
            if startswith(lc, "isa")
                hart_count += 1
                if occursin("rv64gcv", lc) || (occursin("rv64", lc) && occursin("_v", lc))
                    has_v = true
                end
                if occursin("zba", lc)
                    has_zba = true
                end
                if occursin("zbb", lc)
                    has_zbb = true
                end
            end
        end
    catch
        # /proc/cpuinfo not readable — use fallback
        hart_count = 0
    end

    return has_v, has_zba, has_zbb, max(hart_count, 1)
end

function detect_riscv_capabilities() :: RiscvCapabilities
    timestamp = string(Dates.now())

    arch = Sys.ARCH
    is_rv64 = arch === :riscv64
    is_rv32 = arch === :riscv32
    is_riscv = is_rv64 || is_rv32
    xlen = is_rv32 ? 32 : 64

    if is_riscv
        has_v, has_zba, has_zbb, parsed_harts = _parse_riscv_cpuinfo()
        hart_count = parsed_harts > 0 ? parsed_harts : Sys.CPU_THREADS
        # Default cache-line size for most RISC-V SoCs
        cache_line_bytes = 64
        return RiscvCapabilities(true, has_v, has_zba, has_zbb, hart_count, xlen, cache_line_bytes, timestamp)
    else
        return RiscvCapabilities(false, false, false, false, Sys.CPU_THREADS, 64, 64, timestamp)
    end
end

# Module-level singleton; reassigned in NexusMinerQTV.__init__() after precompilation.
RISCV = detect_riscv_capabilities()

function print_riscv_capabilities(io::IO = stdout)
    c = RISCV
    println(io, "┌─────────────────────────────────────────")
    println(io, "│ NexusMinerQTV — RISC-V Capability Report")
    println(io, "├─────────────────────────────────────────")
    println(io, "│  is_riscv         : $(c.is_riscv)")
    println(io, "│  has_v_ext        : $(c.has_v_ext)")
    println(io, "│  has_zba          : $(c.has_zba)")
    println(io, "│  has_zbb          : $(c.has_zbb)")
    println(io, "│  hart_count       : $(c.hart_count)")
    println(io, "│  xlen             : $(c.xlen)")
    println(io, "│  cache_line_bytes : $(c.cache_line_bytes)")
    println(io, "│  detected_at      : $(c.detected_at)")
    println(io, "└─────────────────────────────────────────")
end
