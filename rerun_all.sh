#!/usr/bin/env bash
# =====================================================================================================================
# rerun_all.sh -- rerun EVERY simulation and every derived table / figure of the paper with the new Theta(x=0) convention
#                 ([model] theta_barrier = 0.5 in config.toml; REVIEW.md E3), in parallel, from one command:
#
#       nohup bash rerun_all.sh --yes > rerun.out 2>&1 &         # then: tail -f rerun.out     (or run it inside tmux/screen)
#
# Read before you start:  bash rerun_all.sh --dry-run            (what it will do, the schedule, the expected wall time)
#                         bash rerun_all.sh --smoke              (5-10 min rehearsal of the whole machinery on a COPY of the repo)
#
# STAGES
#   0 preflight   python >= 3.11 + numpy/scipy/matplotlib/TeNPy, free disk, running jobs of this repo, config, the two test suites
#   1 archive     the results of the earlier setting (data/runs, controls, studies/dt_scan + parameter_map, mps, derived, logs) are MOVED
#                 to previous-data/<date>_before_rerun/ (tracked in git).  Nothing is deleted.  Nothing old can leak into the new tables.
#                 (data/studies/onebody_mc, the one-body / Monte Carlo check logs that the paper reads, stays: it is an input.)
#   2 simulate    every simulation, independent of each other, in parallel: a pool that starts the longest job first (list scheduling) and
#                 never uses more than --cores threads: 16 exact runs of data/runs (incl. the new N=5 k=20 and N=6 k=12), the dt study,
#                 the off-resonance control, the 4 MPS validation cases (exact reference + TeNPy MPS), the parameter scans, the Appendix-A
#                 ledger (ljtunnel/parameter_study.py), the full-grid N=2 reference.  All with force=True and the FROZEN tilts of [tilts] (no scans).
#   3 derive      run_records, run_protocol_tables, run_extrapolation, run_quantum, run_cosine_check, run_parameter_choice,
#                 run_mps_tables(step_cost=True) [a timing measurement: alone, on the idle machine], run_theta_effect (old vs new)
#   4 check       run_check_rerun: every expected file exists, is newer than the start, has theta_barrier = 0.5, the config tilt and the
#                 independent <O>_gs anchors; no duplicates.  Then the two test suites again.
#   5 paper       paper/scripts/make_numbers.py (claims check), figs.py, and pdflatex/bibtex if installed.
#
# RESUME   A job that finished (exit 0) leaves data/rerun/state/done/<job>; after a crash / power loss / ^C run the same command with
#          --resume: finished jobs are skipped, unfinished ones restart from their beginning (a job writes its file only at its end).
#          The config.toml must not change between the start and a resume (its hash is checked).  Do not edit code while it runs.
#
# OUTPUT   data/runs, data/studies/dt_scan, data/controls, data/mps (+ed_ref), data/derived, data/studies/parameter_map/tier1_parameter_map.txt,
#          paper/numbers.tex, tab_*.tex, figures/;  logs: data/logs (the simulations' own), data/rerun/joblogs (one file per job and step),
#          data/rerun/summary.txt (versions, timings, results of the checks).  Commit them:  git add -A data paper && git commit
#
# OPTIONS  --cores N        threads the pool may use (default: physical cores; hyper-threads do not help this code)
#          --yes            do not ask before archiving the old results
#          --resume         continue an interrupted rerun (no archive, no tests)
#          --post-only      skip the simulations (stages 3-5 only), e.g. after fixing something by hand
#          --no-paper       stop after the checks (no make_numbers / figs / LaTeX)
#          --no-tests       skip the test suites
#          --ignore-running do not refuse to start because of other python processes that look like simulations of this repository
#          --dry-run        print the plan and the estimated schedule, change nothing
#          --smoke          rehearsal on a temporary copy of the repo with the small jobs only (N <= 3, k <= 16, one MPS case)
#          --rehearse KIND  the same on a copy, with KIND = smoke | medium (medium: everything except the four longest jobs, ~2 h)
#          --speed X        machine speed relative to the audit core, for the time estimate only (default 1.3)
#          --python PATH    interpreter (default: the first of python3, python that is >= 3.11)
#          -h, --help
# =====================================================================================================================
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 2
SELF="$ROOT/$(basename "${BASH_SOURCE[0]}")"
export ROOT

RUNDIR="$ROOT/data/rerun"
STATE="$RUNDIR/state"
JOBLOGS="$RUNDIR/joblogs"
ARCHIVE="$ROOT/previous-data/$(date +%Y%m%d_%H%M%S)_before_rerun"      # a fresh start archives into a new directory; a resume reads it back

# ------------------------------------------------------------------------------------------------ internal: run one job
# `bash rerun_all.sh --run-job NAME THREADS CALL` is what the pool starts for every job (own process group, own log).
if [ "${1:-}" = "--run-job" ]; then
    export STATE JOBLOGS
    name=$2; thr=$3; call=$4
    mkdir -p "$STATE"/{status,done,start,end,pids} "$JOBLOGS"
    logf="$JOBLOGS/job_${name}.log"
    [ -s "$logf" ] && mv "$logf" "$logf.prev.$(date +%Y%m%d_%H%M%S)"
    rm -f "$STATE/status/$name" "$STATE/done/$name" "$STATE/end/$name"
    date +%s > "$STATE/start/$name"
    echo $$ > "$STATE/pids/$name"
    export OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr VECLIB_MAXIMUM_THREADS=$thr
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    {
        echo "# job $name   threads $thr   host $(hostname)   start $(date -Is)"
        echo "# call: $call"
    } > "$logf"
    nice -n 10 "$PYTHON" -u -c "from run import *; $call" >> "$logf" 2>&1
    rc=$?
    echo "# exit $rc   end $(date -Is)" >> "$logf"
    date +%s > "$STATE/end/$name"
    rm -f "$STATE/pids/$name"
    [ "$rc" -eq 0 ] && : > "$STATE/done/$name"
    echo "$rc" > "$STATE/status/$name"
    exit "$rc"
fi

# ------------------------------------------------------------------------------------------------ options
IGNORE_RUNNING=0; CORES=""; ASSUME_YES=0; RESUME=0; POST_ONLY=0; NO_PAPER=0; NO_TESTS=0; DRY=0; SMOKE=0; SUBSET=""; SPEED=1.3; PYTHON="${PYTHON:-}"
usage() { awk '/^# =====/ {c++; if (c == 2) exit; next} c == 1 {sub(/^# ?/, ""); print}' "$SELF"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --cores) CORES=$2; shift ;;
        --yes|-y) ASSUME_YES=1 ;;
        --resume) RESUME=1 ;;
        --post-only) POST_ONLY=1; RESUME=1 ;;
        --no-paper) NO_PAPER=1 ;;
        --no-tests) NO_TESTS=1 ;;
        --ignore-running) IGNORE_RUNNING=1 ;;
        --dry-run) DRY=1 ;;
        --smoke) SMOKE=1; SUBSET=smoke ;;
        --rehearse) SMOKE=1; SUBSET=$2; shift ;;
        --smoke-inner) SMOKE=2 ;;            # internal: the copy re-executes itself with this
        --subset) SUBSET=$2; shift ;;        # internal
        --speed) SPEED=$2; shift ;;
        --python) PYTHON=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

# ------------------------------------------------------------------------------------------------ helpers
MASTER_LOG="$JOBLOGS/rerun_all.log"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$MASTER_LOG" 2>/dev/null || printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
hms() { local s=$1; if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60)); else printf '%dm%02ds' $((s / 60)) $((s % 60)); fi; }

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 3 ]; }; then
    echo "bash >= 4.3 is required (found $BASH_VERSION)" >&2; exit 2
fi

find_python() {
    local cand
    for cand in "$PYTHON" python3 python; do
        [ -n "$cand" ] || continue
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON=$(command -v "$cand"); return 0
        fi
    done
    return 1
}
find_python || { echo "no python >= 3.11 found (tomllib is needed); use --python PATH" >&2; exit 2; }
export PYTHON

live_job_pid() {   # live_job_pid NAME: the pid of a still-running job of this rerun (pid file + command line check), else status 1
    local p; p=$(cat "$STATE/pids/$1" 2>/dev/null) || return 1
    [ -n "$p" ] || return 1
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--run-job $1 " && echo "$p"
}

physical_cores() {
    local n=""
    if command -v lscpu >/dev/null 2>&1; then n=$(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l); fi
    if [ -z "$n" ] || [ "$n" -lt 1 ]; then n=$(nproc 2>/dev/null || echo 4); fi
    echo "$n"
}
[ -n "$CORES" ] || CORES=$(physical_cores)
case "$CORES" in ''|*[!0-9]*) echo "--cores needs a number" >&2; exit 2 ;; esac
[ "$CORES" -ge 1 ] || { echo "--cores must be >= 1" >&2; exit 2; }

# ------------------------------------------------------------------------------------------------ smoke: rehearse on a copy
if [ "$SMOKE" -eq 1 ]; then
    case "$SUBSET" in smoke|medium) ;; *) echo "--rehearse needs smoke or medium" >&2; exit 2 ;; esac
    TMP=$(mktemp -d "${TMPDIR:-/tmp}/lj-rehearsal.XXXXXX") || exit 2
    echo "$SUBSET rehearsal in a COPY of the repository: $TMP  (your repository is not touched; delete the copy when you are done)"
    tar --exclude=./.git --exclude=./data/ckpt --exclude=./data/rerun -cf - -C "$ROOT" . | tar -xf - -C "$TMP" || exit 2
    args=(--smoke-inner --subset "$SUBSET" --yes --cores "$CORES" --python "$PYTHON" --speed "$SPEED")
    [ "$SUBSET" = smoke ] && args+=(--no-paper)
    [ "$NO_TESTS" -eq 1 ] && args+=(--no-tests)
    exec bash "$TMP/rerun_all.sh" "${args[@]}"
fi

if [ "$DRY" -eq 1 ]; then          # a dry run writes nothing into the repository
    RUNDIR=$(mktemp -d "${TMPDIR:-/tmp}/lj-dry.XXXXXX"); STATE="$RUNDIR/state"; JOBLOGS="$RUNDIR/joblogs"; MASTER_LOG="$JOBLOGS/rerun_all.log"
fi
export STATE JOBLOGS
mkdir -p "$STATE"/{status,done,start,end,pids} "$JOBLOGS" 2>/dev/null

# ------------------------------------------------------------------------------------------------ the job table
JOBS_TSV="$STATE/jobs.tsv"
"$PYTHON" run.py jobs $SUBSET > "$JOBS_TSV" || die "run.py jobs failed"
[ -s "$JOBS_TSV" ] || die "empty job table"
# With few cores the two longest jobs (N=6 k=12, N=5 k=20; ~12 h each on 2 threads) would occupy the machine for half a day while the
# 25 short jobs wait.  They go to the END of the queue instead: the short jobs finish first and their data can be committed at once.
if [ "$CORES" -lt 8 ]; then
    { grep -vE '^(N6_k12|N5_k20)'"$(printf '\t')" "$JOBS_TSV"; grep -E '^(N6_k12|N5_k20)'"$(printf '\t')" "$JOBS_TSV"; } > "$JOBS_TSV.tmp" && mv "$JOBS_TSV.tmp" "$JOBS_TSV"
fi

plan() {   # estimated schedule of the pool (list scheduling, longest first, never more than CORES threads)
    "$PYTHON" - "$CORES" "$SPEED" "$JOBS_TSV" <<'PYEOF'
import sys
cores, speed = int(sys.argv[1]), float(sys.argv[2])
jobs = [l.rstrip('\n').split('\t') for l in open(sys.argv[3]) if l.strip()]
jobs = [(n, min(int(t), cores), float(h) / speed, c) for n, t, h, c in jobs]
free, t, running, pending, rows = cores, 0.0, [], list(jobs), []
while pending or running:
    again = True
    while again:
        again = False
        for j in pending:
            if j[1] <= free:
                pending.remove(j); free -= j[1]; running.append((t + j[2], j)); rows.append((j[0], j[1], t, t + j[2])); again = True; break
    if not running:
        break
    running.sort(key=lambda x: x[0]); t, j = running.pop(0); free += j[1]
print(f"{'job':<22}{'thr':>4}{'start':>9}{'end':>9}   (hours, estimate)")
for n, thr, a, b in rows:
    print(f"{n:<22}{thr:>4}{a:9.2f}{b:9.2f}")
cph = sum(j[1] * j[2] for j in jobs)
print(f"{len(jobs)} jobs, {cph:.1f} core-hours, estimated wall time of the pool {t:.1f} h on {cores} cores at speed x{speed:g} "
      f"(the longest jobs set it: extra cores beyond what they and the short jobs can use do not shorten it)")
PYEOF
}

# ------------------------------------------------------------------------------------------------ preflight
preflight() {
    log "preflight: python $("$PYTHON" -c 'import sys; print(sys.version.split()[0])')  at $PYTHON;  bash $BASH_VERSION;  $CORES cores for the pool"
    "$PYTHON" - <<'PYEOF' || return 1
import importlib, sys
need = {'numpy': True, 'scipy': True, 'tenpy': True, 'matplotlib': False}
bad = False
for m, hard in need.items():
    try:
        mod = importlib.import_module(m)
        print(f"   {m} {getattr(mod, '__version__', '?')}")
    except ImportError as e:
        print(f"   MISSING {m}: {e}" + ('' if hard else '   (only the paper figures need it; use --no-paper)'))
        bad = bad or hard
sys.exit(1 if bad else 0)
PYEOF
    local th; th=$("$PYTHON" -c "import run; print(run.theta_barrier())" 2>&1 | tail -n 1)
    case "$th" in [0-9]*) ;; *) die "cannot import run.py: $th" ;; esac
    log "preflight: config theta_barrier = $th"
    [ "$th" = "0.5" ] || die "config.toml [model] theta_barrier is $th, expected 0.5 for this rerun"
    [ -f data/studies/onebody_mc/mc_gamma_ledger.txt ] || die "data/studies/onebody_mc/mc_gamma_ledger.txt is missing: it is an input of the paper scripts (README there)"
    local free; free=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
    [ "${free:-0}" -ge 5 ] || die "less than 5 GB free on the disk of $ROOT"
    log "preflight: ${free} GB free"
    # (1) jobs of THIS rerun that are still alive (the master was killed): resume adopts them, nothing else may be started twice
    local own=" " f nm lp
    for f in "$STATE"/pids/*; do
        [ -e "$f" ] || continue
        nm=$(basename "$f"); lp=$(live_job_pid "$nm") || continue
        own="$own$lp "
        log "preflight: job $nm of this rerun is still running (pid $lp): --resume adopts it"
    done
    # (2) simulations of this repository started by somebody else (e.g. the earlier N=5 k=20 / N=6 jobs of the old convention):
    #     any python that holds a log file of data/logs open, or whose working directory is in the repository and whose command line
    #     says it runs this code (the full command line is matched; it is shortened only for display)
    local pid cwd cmd exe found="" other="" pg
    for pid in $( { find /proc/[0-9]*/fd -maxdepth 1 -lname "$ROOT/data/logs/*" 2>/dev/null | sed 's|/proc/\([0-9]*\)/.*|\1|'; pgrep -f 'python' 2>/dev/null; } | sort -u); do
        [ "$pid" = "$$" ] && continue
        [ -r "/proc/$pid/cmdline" ] || continue
        pg=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $3}')
        case "$own" in *" $pid "*|*" $pg "*) continue ;; esac
        exe=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -n1); exe=${exe##*/}
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
        holds=$(find "/proc/$pid/fd" -maxdepth 1 -lname "$ROOT/data/logs/*" 2>/dev/null | head -n1)
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        case "$exe" in python*|ipython*|Python*) ;; *) [ -n "$holds" ] || continue ;; esac
        if [ -n "$holds" ]; then found="$found ${pid}[${cmd:0:100}]"; continue; fi
        case "$cwd" in "$ROOT"|"$ROOT"/*) ;; *) continue ;; esac
        case "$cmd" in
            *run_dynamics*|*run_production*|*dynamics_run*|*"from run import"*|*run.py*|*ljtunnel*) found="$found ${pid}[${cmd:0:100}]" ;;
            *) other="$other ${pid}[${cmd:0:100}]" ;;
        esac
    done
    [ -n "$other" ] && log "preflight: note: other python processes with this repository as working directory (an open Spyder/IPython session is fine):$other"
    if [ -n "$found" ]; then
        log "preflight: simulations of this repository are already running:$found"
        if [ "$IGNORE_RUNNING" -eq 1 ]; then log "preflight: --ignore-running: starting anyway"
        else die "stop them first (they would compete for the cores, and their results use the OLD convention); a false positive: --ignore-running"; fi
    fi
    local load; load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
    "$PYTHON" -c "import sys; sys.exit(0 if float('$load') < 0.25 * $CORES else 1)" || log "preflight: WARNING: the 1-minute load average is $load: something else is using this machine (it slows the pool and makes the timing step unreliable)"
    if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse >/dev/null 2>&1; then
        local dirty; dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l)
        log "preflight: git $(git -C "$ROOT" rev-parse --short HEAD), $dirty modified tracked files"
    fi
    command -v pdflatex >/dev/null 2>&1 && log "preflight: pdflatex found" || log "preflight: pdflatex NOT found: the PDF will not be built (numbers.tex, tables and figures are)"
    return 0
}

run_tests() {
    log "tests: tests/test_numerics.py (~15 s) and tests/test_mps.py (~1 min)"
    ( export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; "$PYTHON" tests/test_numerics.py && "$PYTHON" tests/test_mps.py ) \
        > "$JOBLOGS/tests_$1.log" 2>&1 || { tail -n 25 "$JOBLOGS/tests_$1.log"; return 1; }
    grep -h '^TESTS' "$JOBLOGS/tests_$1.log" | tr '\n' ' '; echo
}

# ------------------------------------------------------------------------------------------------ archive
archive_old() {   # archive_old [again]: moves the old results away; `again` (a resume after an interrupted archive) finishes the job
    if [ -e "$ARCHIVE" ] && [ "${1:-}" != again ]; then die "$ARCHIVE exists already (an earlier rerun?).  Use --resume to continue it, or move that directory away."; fi
    mkdir -p "$ARCHIVE"
    local moved=0 src
    mvx() {   # mvx SRC DST : move a file or directory if it exists
        [ -e "$1" ] || return 0
        mkdir -p "$(dirname "$2")" && mv "$1" "$2" && moved=$((moved + 1))
    }
    mvx data/runs "$ARCHIVE/runs"
    mvx data/studies/dt_scan "$ARCHIVE/studies/dt_scan"
    mvx data/studies/parameter_map "$ARCHIVE/studies/parameter_map"
    mvx data/controls "$ARCHIVE/controls"
    mkdir -p "$ARCHIVE/mps"
    for src in data/mps/*.npz; do [ -e "$src" ] && mvx "$src" "$ARCHIVE/mps/$(basename "$src")"; done
    mvx data/mps/ed_ref "$ARCHIVE/mps/ed_ref"
    mkdir -p "$ARCHIVE/derived/tn_tables"
    for src in data/derived/*.npz data/derived/*.csv; do [ -e "$src" ] && mvx "$src" "$ARCHIVE/derived/$(basename "$src")"; done
    for src in data/derived/tn_tables/*.csv; do [ -e "$src" ] && mvx "$src" "$ARCHIVE/derived/tn_tables/$(basename "$src")"; done
    mkdir -p "$ARCHIVE/logs"
    for src in data/logs/*.log; do [ -e "$src" ] && mvx "$src" "$ARCHIVE/logs/$(basename "$src")"; done
    {
        echo "Results made with [model] theta_barrier = 0 (the grid column x = 0 counted as left), moved here by rerun_all.sh on $(date -Is)."
        echo "git: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown); they are also in the git history (git show <rev>:data/...)."
        echo "run.py: run_theta_effect() compares data/runs with ${ARCHIVE#"$ROOT"/}/runs."
    } > "$ARCHIVE/ARCHIVE.txt"
    mkdir -p data/runs data/studies/dt_scan data/studies/parameter_map data/controls data/mps/ed_ref data/derived/tn_tables data/logs
    : > "$STATE/archived"
    log "archive: $moved items moved to ${ARCHIVE#"$ROOT"/} ($(du -sh "$ARCHIVE" | cut -f1))"
}

# ------------------------------------------------------------------------------------------------ the pool
J_NAME=(); J_THR=(); J_HRS=(); J_CALL=(); J_ST=(); J_PID=()
POOL_FAILED=0; FAILED_NAMES=""; CFG_CHANGED=0

stop_children() {
    local i
    for ((i = 0; i < ${#J_ST[@]}; i++)); do
        if [ "${J_ST[i]}" = running ]; then kill -TERM -- "-${J_PID[i]}" 2>/dev/null || kill -TERM "${J_PID[i]}" 2>/dev/null; fi
    done
}

run_pool() {
    local i n used=0 running pending nfail last_beat now t0 lp
    mapfile -t lines < "$JOBS_TSV"
    n=${#lines[@]}
    for ((i = 0; i < n; i++)); do
        IFS=$'\t' read -r "J_NAME[$i]" "J_THR[$i]" "J_HRS[$i]" "J_CALL[$i]" <<< "${lines[i]}"
        (( J_THR[i] > CORES )) && J_THR[i]=$CORES
        J_PID[i]=0
        if [ -e "$STATE/done/${J_NAME[i]}" ]; then J_ST[i]=skipped
        elif lp=$(live_job_pid "${J_NAME[i]}"); then J_ST[i]=running; J_PID[i]=$lp; used=$((used + J_THR[i])); log "adopt  ${J_NAME[i]}  (still running from the earlier invocation, pid $lp)"
        else J_ST[i]=pending; fi
    done
    local nskip=0; for ((i = 0; i < n; i++)); do [ "${J_ST[i]}" = skipped ] && nskip=$((nskip + 1)); done
    log "pool: $n jobs ($nskip already done), $CORES cores; the longest start first"
    t0=$(date +%s); last_beat=$t0
    while :; do
        for ((i = 0; i < n; i++)); do               # start every pending job that fits (list scheduling, in order of decreasing cost)
            [ "${J_ST[i]}" = pending ] || continue
            (( used + J_THR[i] <= CORES )) || continue
            if [ "$(sha256sum config.toml | cut -d' ' -f1)" != "$(cat "$STATE/config.sha256")" ]; then
                [ "$CFG_CHANGED" -eq 1 ] || log "ERROR: config.toml changed while the rerun is running: no new job is started (the jobs already running finish)"
                CFG_CHANGED=1; break
            fi
            if command -v setsid >/dev/null 2>&1; then
                setsid bash "$SELF" --run-job "${J_NAME[i]}" "${J_THR[i]}" "${J_CALL[i]}" > /dev/null 2>&1 < /dev/null &
            else
                bash "$SELF" --run-job "${J_NAME[i]}" "${J_THR[i]}" "${J_CALL[i]}" > /dev/null 2>&1 < /dev/null &
            fi
            J_PID[i]=$!; J_ST[i]=running; used=$((used + J_THR[i]))
            log "start  ${J_NAME[i]}  (${J_THR[i]} threads, est ${J_HRS[i]} h)   [$used/$CORES threads in use]"
        done
        running=0; pending=0
        for ((i = 0; i < n; i++)); do
            case "${J_ST[i]}" in running) running=$((running + 1)) ;; pending) pending=$((pending + 1)) ;; esac
        done
        if [ "$running" -eq 0 ]; then break; fi      # nothing running: all done (a pending job always fits when nothing runs)
        sleep 5
        for ((i = 0; i < n; i++)); do                # reap
            [ "${J_ST[i]}" = running ] || continue
            local rc=""
            if [ -f "$STATE/status/${J_NAME[i]}" ]; then rc=$(cat "$STATE/status/${J_NAME[i]}")
            elif ! kill -0 "${J_PID[i]}" 2>/dev/null; then rc=137; fi
            [ -n "$rc" ] || continue
            used=$((used - J_THR[i]))
            local el=0; [ -f "$STATE/start/${J_NAME[i]}" ] && [ -f "$STATE/end/${J_NAME[i]}" ] && el=$(( $(cat "$STATE/end/${J_NAME[i]}") - $(cat "$STATE/start/${J_NAME[i]}") ))
            if [ "$rc" -eq 0 ]; then J_ST[i]="done"; log "done   ${J_NAME[i]}  in $(hms "$el")"
            else J_ST[i]=failed; FAILED_NAMES="$FAILED_NAMES ${J_NAME[i]}"; log "FAILED ${J_NAME[i]}  (exit $rc after $(hms "$el"); see ${JOBLOGS#"$ROOT"/}/job_${J_NAME[i]}.log)"; tail -n 8 "$JOBLOGS/job_${J_NAME[i]}.log" 2>/dev/null | sed 's/^/         | /'; fi
        done
        now=$(date +%s)
        if [ $((now - last_beat)) -ge 1200 ]; then
            last_beat=$now
            local msg=""
            for ((i = 0; i < n; i++)); do
                [ "${J_ST[i]}" = running ] || continue
                msg="$msg"$'\n'"         ${J_NAME[i]}: $(tail -n 1 "$JOBLOGS/job_${J_NAME[i]}.log" 2>/dev/null | cut -c1-110)"
            done
            log "heartbeat: elapsed $(hms $((now - t0))), $running running, $pending waiting$msg"
        fi
    done
    nfail=0; for ((i = 0; i < n; i++)); do [ "${J_ST[i]}" = failed ] && nfail=$((nfail + 1)); done
    for ((i = 0; i < n; i++)); do [ "${J_ST[i]}" = pending ] && { nfail=$((nfail + 1)); FAILED_NAMES="$FAILED_NAMES ${J_NAME[i]}(not started)"; }; done
    POOL_FAILED=$nfail
    log "pool finished in $(hms $(($(date +%s) - t0))): $((n - nfail)) of $n jobs done, $nfail not"
}

# ------------------------------------------------------------------------------------------------ post-processing steps
POST_FAILED=0; PDF_STALE=0
post() {   # post NAME PYTHON_STATEMENT [threads]: one analysis step, own log; a failing step is recorded, not fatal
    local name=$1 call=$2 thr=${3:-$CORES} t0 rc logf="$JOBLOGS/post_$1.log"
    t0=$(date +%s)
    if [ "$thr" = default ]; then
        ( export PYTHONUNBUFFERED=1 MPLBACKEND=Agg; "$PYTHON" -u -c "from run import *; $call" ) > "$logf" 2>&1; rc=$?
    else
        ( export OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr PYTHONUNBUFFERED=1 MPLBACKEND=Agg
          "$PYTHON" -u -c "from run import *; $call" ) > "$logf" 2>&1; rc=$?
    fi
    if [ "$rc" -eq 0 ]; then log "step   $name  ok ($(hms $(($(date +%s) - t0))))"
    else POST_FAILED=$((POST_FAILED + 1)); log "step   $name  FAILED (exit $rc; ${logf#"$ROOT"/})"; tail -n 8 "$logf" | sed 's/^/         | /'; fi
    return "$rc"
}

paper_build() {
    local rc=0
    ( cd paper || exit 1
      "$PYTHON" -I scripts/make_numbers.py > "$JOBLOGS/paper_make_numbers.log" 2>&1 ) || { log "paper: make_numbers.py FAILED (${JOBLOGS#"$ROOT"/}/paper_make_numbers.log)"; tail -n 8 "$JOBLOGS/paper_make_numbers.log" | sed 's/^/         | /'; return 1; }
    grep -h 'hand-written claims confirmed' "$JOBLOGS/paper_make_numbers.log" | sed 's/^/paper: /' | tee -a "$MASTER_LOG"
    if grep -q 'CLAIM FAILS' "$JOBLOGS/paper_make_numbers.log"; then
        log "paper: CLAIMS THAT THE NEW DATA CONTRADICT (edit the text):"; grep 'CLAIM FAILS' "$JOBLOGS/paper_make_numbers.log" | sed 's/^/         | /' | tee -a "$MASTER_LOG"; rc=1
    fi
    ( cd paper || exit 1; "$PYTHON" -I scripts/figs.py > "$JOBLOGS/paper_figs.log" 2>&1 ) || { log "paper: figs.py FAILED (${JOBLOGS#"$ROOT"/}/paper_figs.log)"; tail -n 8 "$JOBLOGS/paper_figs.log" | sed 's/^/         | /'; return 1; }
    log "paper: numbers.tex, tab_*.tex and the figures rebuilt"
    local pend; pend=$(grep -c '\\pending{' paper/numbers.tex paper/tab_*.tex 2>/dev/null | awk -F: '$2 > 0 {printf "%s(%s) ", $1, $2}')
    [ -n "$pend" ] && { log "paper: \\pending markers left in: $pend"; rc=1; } || log "paper: no \\pending marker left"
    local undef; undef=$("$PYTHON" - <<'PYEOF'
import glob, re
used = set()
for f in glob.glob('paper/*.tex'):
    if not f.endswith('numbers.tex'):
        used |= set(re.findall(r'\\num\{([^}]+)\}', open(f, errors='replace').read()))
defined = set(re.findall(r'\\defnum\{([^}]+)\}', open('paper/numbers.tex').read()))
print(' '.join(sorted(used - defined)))
PYEOF
)
    if [ -n "$undef" ]; then log "paper: \\num keys used in the text but not defined by numbers.tex (they print as red ??key): $undef"; rc=1; else log "paper: every \\num key of the text is defined"; fi
    if command -v pdflatex >/dev/null 2>&1; then
        ( cd paper || exit 1
          pdflatex -interaction=nonstopmode main > "$JOBLOGS/paper_latex.log" 2>&1
          bibtex main >> "$JOBLOGS/paper_latex.log" 2>&1
          pdflatex -interaction=nonstopmode main >> "$JOBLOGS/paper_latex.log" 2>&1
          pdflatex -interaction=nonstopmode main >> "$JOBLOGS/paper_latex.log" 2>&1 )
        if grep -qE 'Undefined control sequence|Reference .* undefined|Citation .* undefined|^! ' "$JOBLOGS/paper_latex.log" paper/main.log 2>/dev/null; then
            log "paper: LaTeX reported problems: grep 'Undefined\|^!' paper/main.log"; rc=1
        else log "paper: paper/main.pdf built without undefined references"; fi
    else log "paper: pdflatex is not installed: paper/main.pdf is the OLD one; compile paper/main.tex elsewhere"; PDF_STALE=1; fi
    return $rc
}

# ------------------------------------------------------------------------------------------------ main
lock() {
    if [ -d "$STATE/lock" ] && kill -0 "$(cat "$STATE/lock/pid" 2>/dev/null)" 2>/dev/null; then
        die "another rerun_all.sh (pid $(cat "$STATE/lock/pid")) is running in this repository"
    fi
    rm -rf "$STATE/lock"; mkdir "$STATE/lock" && echo $$ > "$STATE/lock/pid"
}
cleanup() { stop_children; rm -rf "$STATE/lock"; }
trap '' HUP      # a closed terminal / dropped ssh must not end a two-day run (the jobs are in their own sessions as well)

if [ "$DRY" -eq 1 ]; then
    echo "== rerun_all.sh --dry-run =="
    preflight || exit 1
    echo; echo "would move to ${ARCHIVE#"$ROOT"/}:"; for d in data/runs data/controls data/studies/dt_scan data/studies/parameter_map data/mps data/derived data/logs; do [ -e "$d" ] && echo "   $d  ($(du -sh "$d" | cut -f1))"; done
    echo; plan
    echo; echo "then: run_records, run_protocol_tables, run_extrapolation, run_quantum, run_cosine_check, run_parameter_choice, run_mps_tables(step_cost=True), run_theta_effect, checks, paper build"
    echo "start for real:  nohup bash rerun_all.sh --yes > rerun.out 2>&1 &"
    exit 0
fi

lock
trap 'log "interrupted: stopping the jobs (run again with --resume to continue)"; cleanup; exit 130' INT TERM
trap 'rm -rf "$STATE/lock"' EXIT      # installed only now: a refused second invocation must not delete the running one's lock
T_START=$(date +%s)
if [ "$RESUME" -eq 1 ]; then
    [ -f "$STATE/started" ] || die "nothing to resume (no $STATE/started)"
    [ "$(sha256sum config.toml | cut -d' ' -f1)" = "$(cat "$STATE/config.sha256")" ] || die "config.toml changed since the rerun started; a resume would mix parameter sets"
    T_START=$(cat "$STATE/started")
    [ -f "$STATE/archive_path" ] && ARCHIVE=$(cat "$STATE/archive_path")
    log "== resume (started $(date -d "@$T_START" '+%F %T' 2>/dev/null || echo "$T_START")) =="
    preflight || die "preflight failed"
    if [ ! -f "$STATE/archived" ] && [ "$POST_ONLY" -eq 0 ]; then log "the archive step had not finished: completing it"; archive_old again; fi
else
    [ -f "$STATE/started" ] && die "$STATE/started exists: an earlier rerun was started here.  --resume continues it; to start a new one: rm -rf data/rerun/state (scratch), and move the record of that rerun (data/rerun/joblogs, summary.txt, paper_build) into previous-data/."
    [ -e "$RUNDIR/summary.txt" ] && die "data/rerun holds the record of an earlier rerun (summary.txt, joblogs/): move it into previous-data/<name>/ first, so that the new logs do not mix with it."
    log "== rerun_all.sh: Theta(x=0) rerun, $(date '+%F %T') =="
    preflight || die "preflight failed"
    if [ "$NO_TESTS" -eq 0 ]; then run_tests pre || die "tests failed BEFORE any change: fix the environment first"; fi
    log "plan:"; plan | sed 's/^/   /' | tee -a "$MASTER_LOG"
    if [ "$ASSUME_YES" -eq 0 ]; then
        printf 'The old results will be MOVED to %s and ~%s core-hours of simulations started.  Continue? [y/N] ' "${ARCHIVE#"$ROOT"/}" \
            "$(awk -F'\t' '{s += $2 * $3} END {printf "%d", s}' "$JOBS_TSV")"
        read -r ans < /dev/tty || ans=n
        [ "$ans" = y ] || [ "$ans" = Y ] || die "aborted by the user"
    fi
    date +%s > "$STATE/started"; T_START=$(cat "$STATE/started")      # state first: an interrupted archive step can then be resumed
    sha256sum config.toml | cut -d' ' -f1 > "$STATE/config.sha256"
    echo "$ARCHIVE" > "$STATE/archive_path"
    archive_old
fi

# who and what, for the summary
{
    echo "rerun_all.sh   started $(date -d "@$T_START" -Is 2>/dev/null || echo "$T_START")   host $(hostname)   cores used by the pool: $CORES"
    echo "git            $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)  ($(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) modified tracked files)"
    echo "config sha256  $(cat "$STATE/config.sha256")"
    "$PYTHON" -c "import sys, numpy, scipy, tenpy; print('python', sys.version.split()[0], ' numpy', numpy.__version__, ' scipy', scipy.__version__, ' tenpy', tenpy.__version__)"
    echo "cpu            $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')   ($(nproc 2>/dev/null) logical, $(physical_cores) physical)"
} > "$RUNDIR/summary.txt"

if [ "$POST_ONLY" -eq 0 ]; then
    run_pool
    if [ "$POOL_FAILED" -gt 0 ]; then
        log "$POOL_FAILED job(s) failed or did not run:$FAILED_NAMES"
        log "Fix the cause (see data/rerun/joblogs/job_<name>.log), then:  bash rerun_all.sh --resume   (finished jobs are kept)"
        exit 1
    fi
fi

log "== derive =="
post records           "run_records()"                       || true
post protocol_tables   "run_protocol_tables()"               || true
post extrapolation     "run_extrapolation()"                 || true
post quantum           "run_quantum()"                       || true
post cosine_check      "run_cosine_check()"                  || true
if [ "$SUBSET" != smoke ]; then
    post parameter_choice "run_parameter_choice()"           || true
    log "step   mps_tables with the timing measurement (alone on the idle machine, ~5-10 min)"
    post mps_tables      "run_mps_tables(step_cost=True)" default || true
fi
post theta_effect      "run_theta_effect()"                  || true

log "== check =="
CHECK_ARGS=(check); [ -n "$SUBSET" ] && CHECK_ARGS+=("$SUBSET")
"$PYTHON" run.py "${CHECK_ARGS[@]}" "$T_START" > "$JOBLOGS/check.log" 2>&1; CHECK_RC=$?
tail -n 20 "$JOBLOGS/check.log" | sed 's/^/   /' | tee -a "$MASTER_LOG"
[ "$NO_TESTS" -eq 0 ] && [ "$SMOKE" -ne 2 ] && { run_tests post || { log "tests failed AFTER the rerun (see ${JOBLOGS#"$ROOT"/}/tests_post.log)"; POST_FAILED=$((POST_FAILED + 1)); }; }

PAPER_RC=0
if [ "$NO_PAPER" -eq 0 ] && [ "$SUBSET" != smoke ]; then
    log "== paper =="
    paper_build || PAPER_RC=1
fi

{
    echo
    echo "finished       $(date -Is)   (elapsed $(hms $(($(date +%s) - T_START))))"
    echo "jobs           job  threads  seconds  exit"
    for f in "$STATE"/status/*; do
        [ -e "$f" ] || continue
        nm=$(basename "$f"); s=0; [ -f "$STATE/start/$nm" ] && [ -f "$STATE/end/$nm" ] && s=$(( $(cat "$STATE/end/$nm") - $(cat "$STATE/start/$nm") ))
        printf '               %-22s %6s  %7s  %s\n' "$nm" "$(grep -m1 "^# job" "$JOBLOGS/job_$nm.log" 2>/dev/null | sed 's/.*threads \([0-9]*\).*/\1/')" "$s" "$(cat "$f")"
    done
    echo "check          exit $CHECK_RC;  post-processing steps failed: $POST_FAILED;  paper: $([ $PAPER_RC -eq 0 ] && echo ok || echo 'see log')"
} >> "$RUNDIR/summary.txt"

log "== summary: ${RUNDIR#"$ROOT"/}/summary.txt =="
if [ "$CHECK_RC" -eq 0 ] && [ "$POST_FAILED" -eq 0 ] && [ "$PAPER_RC" -eq 0 ]; then
    log "ALL DONE in $(hms $(($(date +%s) - T_START)))."
    [ "$PDF_STALE" -eq 1 ] && log "NOTE: paper/main.pdf was NOT rebuilt (no pdflatex): numbers.tex, the tables and the figures are new, the PDF is old."
    [ "$SMOKE" -eq 2 ] && log "REHEARSAL ($SUBSET) OK: the machinery works here.  Start the real run:  nohup bash rerun_all.sh --yes > rerun.out 2>&1 &   (this copy is in $ROOT)"
    [ "$SMOKE" -ne 2 ] && log "Commit:  git add -A data paper && git commit -m 'Rerun with the weight-1/2 Theta(x=0) convention' && git push"
    exit 0
fi
log "FINISHED WITH PROBLEMS (check: $CHECK_RC, steps failed: $POST_FAILED, paper: $PAPER_RC).  The data are in place; see the logs named above."
exit 1
