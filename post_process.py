#!/usr/bin/env python
"""
Post-processing for the XBeach batch (replaces cells 0-2 of post_3d.ipynb).
 
Each case gets its own small result file (q_tot.json) written right after XBeach
finishes, so "intermediate saving" is automatic and nothing is lost if a job dies.
The summary CSV is just a concatenation of those files and takes seconds to build.
 
The time average only uses output from T_AVG_START (below) to the end of the run.
Each q_tot.json records the T_AVG_START it was computed with; results computed with a
different value are treated as stale and redone automatically.
 
Modes
-----
One case (called by run_case.sh right after XBeach finishes):
    python post_process.py --case xbeach_cases/run_hm0_1.00_s0_0.0200_ang_250.00
 
Collect every finished case into the summary CSV (safe to run any time, even mid-batch):
    python post_process.py --run_dir xbeach_cases --out q_tot_summary_3d.csv
 
(Re)compute q_tot.json for cases that already have a complete xboutput.nc but no
up-to-date result, then collect. Run only when no XBeach jobs are active:
    python post_process.py --run_dir xbeach_cases --compute_missing --workers 8 --out q_tot_check.csv
 
Write the list of cases that still need running (feed this to the SLURM array):
    python post_process.py --case_list case_list.txt --write_pending pending.txt
"""
import os
 
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
 
import argparse
import glob
import json
import re
import sys
import traceback
 
import numpy as np
 
CASE_GLOB = "run_hm0_*_s0_*_ang_*"
NAME_RE = re.compile(r"run_hm0_([\d\.]+)_s0_([\d\.]+)_ang_([\d\.-]+)")
NC_FILE = "xboutput.nc"
RESULT_FILE = "q_tot.json"
ERROR_FILE = "q_tot_error.json"
MEAN_FILE = "xb_timemean.nc"
LOCK_FILE = ".running"
MEAN_VARS = ["Svsg", "Svbg", "Susg", "Subg"]
# Output before this time (s) is left out of the time average (spin-up).
# Change it only here: results computed with a different value are redone automatically.
T_AVG_START = 600.0
 
COLUMNS = ["hm0", "s0", "tp", "mainang", "q_tot", "folder", "t_avg_start"]
EXIT_BAD_OUTPUT = 3  # xboutput.nc missing, truncated or unreadable -> XBeach should be re-run
 
 
class BadOutputError(RuntimeError):
    """xboutput.nc is missing, incomplete or unreadable (as opposed to a bug in this script)."""
 
 
# ----------------------------------------------------------------------------- helpers
def write_json_atomic(path, obj):
    """Write to a temp file then rename, so a killed job never leaves a half-written file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
 
 
def read_param(params_path, key):
    """Read a numeric value like 'tstop = 3600' from an XBeach params.txt (None if absent)."""
    if not os.path.exists(params_path):
        return None
    with open(params_path) as f:
        for line in f:
            line = line.split("!")[0]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip().lower() == key.lower():
                try:
                    return float(v.split()[0])
                except (ValueError, IndexError):
                    return None
    return None
 
 
def load_current_result(case_dir):
    """q_tot.json contents if it exists AND was computed with the current T_AVG_START, else None."""
    path = os.path.join(case_dir, RESULT_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            res = json.load(f)
    except (OSError, ValueError):
        return None
    return res if res.get("t_avg_start") == T_AVG_START else None
 
 
def list_case_dirs(run_dir, case_list=None):
    if case_list:
        with open(case_list) as f:
            return [line.strip().rstrip("/") for line in f if line.strip()]
    return sorted(d.rstrip("/") for d in glob.glob(os.path.join(run_dir, CASE_GLOB)) if os.path.isdir(d))
 
 
# ----------------------------------------------------------------------------- core computation
def compute_case(case_dir, write_mean=True):
    """
    Same q_tot as the original process_single_case() in post_3d.ipynb, except the time
    average only uses output with globaltime >= T_AVG_START. Also:
      * refuses to process an incomplete run (last output time < tstop), so a job that
        was killed mid-run is re-run instead of silently giving a wrong q_tot
      * reads each variable once and also saves the time-mean fields to xb_timemean.nc
        (a few MB) -- enough to recompute q_tot or redo the transport maps later even
        if the multi-GB xboutput.nc is deleted.
    mean(Svsg + Svbg) == mean(Svsg) + mean(Svbg), so the result is identical.
    """
    import xarray as xr
 
    case_dir = os.path.abspath(case_dir)
    folder = os.path.basename(case_dir)
    match = NAME_RE.search(folder)
    if not match:
        raise ValueError("folder name did not match expected pattern")
    hm0_val, s0_val, ang_val = (float(g) for g in match.groups())
    tp_val = float(np.sqrt(hm0_val / (1.56 * s0_val)))
 
    nc_path = os.path.join(case_dir, NC_FILE)
    if not os.path.exists(nc_path):
        raise BadOutputError(f"{NC_FILE} not found")
 
    params_path = os.path.join(case_dir, "params.txt")
    tstop = read_param(params_path, "tstop")
    tintg = read_param(params_path, "tintg") or 1.0
 
    try:
        ds = xr.open_dataset(nc_path, chunks={"globaltime": 20})
        t = ds["globaltime"].values
        t_end = float(t[-1])
    except Exception as e:
        raise BadOutputError(f"could not read {NC_FILE}: {type(e).__name__}: {e}") from e
    with ds:
        if tstop is not None and t_end < tstop - 1.5 * tintg:
            raise BadOutputError(f"incomplete run: last output at t={t_end:g}s but tstop={tstop:g}s")
        keep = t >= T_AVG_START
        if not keep.any():
            raise BadOutputError(f"no output at or after t={T_AVG_START:g}s")
        window = ds.isel(globaltime=slice(int(np.argmax(keep)), None))
        n_avg = int(window.sizes["globaltime"])
 
        present = [v for v in MEAN_VARS if v in ds]
        means = window[present].mean(dim="globaltime").compute()
        globalx = ds["globalx"].compute()
        globaly = ds["globaly"].compute()
 
    time_avg = means["Svsg"] + means["Svbg"]
    y_mask = (globaly >= 800) & (globaly <= 1200)
    transport_y_filtered = time_avg.where(y_mask, drop=True)
    # 20 evenly-spaced alongshore profiles inside the y-window
    n_profiles = transport_y_filtered.sizes["ny"]
    idx = np.unique(np.linspace(0, n_profiles - 1, 20).round().astype(int))
    profile_1d = transport_y_filtered.isel(ny=idx).mean(dim="ny")
    x_coord_1d = globalx.where(y_mask, drop=True).isel(ny=idx).mean(dim="ny")
    profile_1d = profile_1d.assign_coords(x_coord=("nx", x_coord_1d.data))
    q_tot_val = float(profile_1d.integrate(coord="x_coord").squeeze().item())
 
    if write_mean:
        out = means.astype("float32")
        out["globalx"] = globalx
        out["globaly"] = globaly
        out.attrs.update(
            description="time-mean of xboutput.nc over globaltime >= t_avg_start",
            t_avg_start=T_AVG_START,
            t_end=t_end,
            n_times_averaged=n_avg,
        )
        tmp = os.path.join(case_dir, f"{MEAN_FILE}.tmp.{os.getpid()}")
        out.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in present})
        os.replace(tmp, os.path.join(case_dir, MEAN_FILE))
 
    return {
        "hm0": hm0_val,
        "s0": s0_val,
        "tp": tp_val,
        "mainang": ang_val,
        "q_tot": q_tot_val,
        "folder": folder,
        "t_avg_start": T_AVG_START,
        "t_end": t_end,
        "n_times_averaged": n_avg,
    }
 
 
def process_and_record(case_dir, write_mean=True, delete_nc=False, dask_threads=None):
    """
    Compute one case and write q_tot.json (or q_tot_error.json).
    Returns (code, message): 0 = ok, EXIT_BAD_OUTPUT = output missing/incomplete, 1 = other error.
    """
    if dask_threads:
        import dask
 
        dask.config.set(scheduler="threads", num_workers=dask_threads)
 
    case_dir = os.path.abspath(case_dir)
    folder = os.path.basename(case_dir)
    err_path = os.path.join(case_dir, ERROR_FILE)
    try:
        res = compute_case(case_dir, write_mean=write_mean)
    except Exception as e:
        write_json_atomic(err_path, {
            "folder": folder,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        })
        code = EXIT_BAD_OUTPUT if isinstance(e, BadOutputError) else 1
        return code, f"✗ {folder}: {type(e).__name__}: {e}"
 
    write_json_atomic(os.path.join(case_dir, RESULT_FILE), res)
    if os.path.exists(err_path):
        os.remove(err_path)
    if delete_nc and write_mean and os.path.exists(os.path.join(case_dir, MEAN_FILE)):
        os.remove(os.path.join(case_dir, NC_FILE))
    return 0, f"✓ {folder}: q_tot = {res['q_tot']:.6g}"
 
 
# ----------------------------------------------------------------------------- batch modes
def compute_missing(dirs, workers, write_mean):
    todo = []
    for d in dirs:
        if load_current_result(d) is not None:
            continue
        if os.path.exists(os.path.join(d, LOCK_FILE)):
            print(f"  skipping {os.path.basename(d)} (a job is running it, or was killed: {LOCK_FILE} present)")
            continue
        if os.path.exists(os.path.join(d, NC_FILE)):
            todo.append(d)
    print(f"Computing {len(todo)} case(s) that have {NC_FILE} but no up-to-date {RESULT_FILE} "
          f"(t_avg_start={T_AVG_START:g}s), {workers} worker(s)")
    if not todo:
        return
 
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    threads = max(1, cpus // workers)
 
    if workers == 1:
        for i, d in enumerate(todo, 1):
            _, msg = process_and_record(d, write_mean, dask_threads=threads)
            print(f"[{i}/{len(todo)}] {msg}", flush=True)
        return
 
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
 
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(process_and_record, d, write_mean, False, threads) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _, msg = fut.result()
            except Exception as e:  # worker process crashed (e.g. out of memory)
                msg = f"✗ worker crashed: {type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {msg}", flush=True)
 
 
def collect(dirs, out):
    import pandas as pd
 
    results, errors, pending, stale = [], [], [], 0
    for d in dirs:
        err_path = os.path.join(d, ERROR_FILE)
        res = load_current_result(d)
        if res is not None:
            results.append(res)
            continue
        if os.path.exists(os.path.join(d, RESULT_FILE)):
            stale += 1
        if os.path.exists(err_path):
            with open(err_path) as f:
                errors.append(json.load(f))
        else:
            pending.append(os.path.basename(d))
 
    df = pd.DataFrame(results, columns=COLUMNS)
    tmp = f"{out}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
 
    root, ext = os.path.splitext(out)
    err_out = f"{root}_errors{ext or '.csv'}"
    if errors:
        pd.DataFrame(errors).to_csv(err_out, index=False)
    elif os.path.exists(err_out):
        os.remove(err_out)
 
    print(f"{len(dirs)} cases: {len(results)} done, {len(errors)} with errors, {len(pending)} not run yet "
          f"(t_avg_start={T_AVG_START:g}s)")
    if stale:
        print(f"  {stale} case(s) have an old q_tot.json from a different averaging window (counted as not done)")
    print(f"Wrote {out}" + (f" and {err_out}" if errors else ""))
 
 
def write_pending(dirs, path):
    pending = [d for d in dirs if load_current_result(d) is None]
    with open(path, "w") as f:
        f.write("".join(f"{d}\n" for d in pending))
    print(f"{len(pending)} of {len(dirs)} cases still need running -> {path}")
    if pending:
        print(f"Submit with:  sbatch --array=0-{len(pending) - 1}%5 xbeach_array.sbatch {path}")
 
 
# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", help="process a single case folder -> q_tot.json")
    p.add_argument("--is_done", metavar="CASE",
                   help="exit 0 if CASE has an up-to-date q_tot.json, 1 if not (used by run_case.sh)")
    p.add_argument("--run_dir", default="xbeach_cases", help="folder holding the run_* case folders")
    p.add_argument("--case_list", help="text file of case folders (one per line) to use instead of globbing run_dir")
    p.add_argument("--out", default="q_tot_summary_3d.csv", help="summary CSV to write")
    p.add_argument("--write_pending", metavar="FILE", help="write unfinished case folders to FILE and exit")
    p.add_argument("--compute_missing", action="store_true",
                   help="before collecting, compute q_tot.json for cases with xboutput.nc but no result")
    p.add_argument("--workers", type=int, default=4, help="parallel files for --compute_missing")
    p.add_argument("--no_mean_file", action="store_true", help=f"don't write {MEAN_FILE}")
    p.add_argument("--delete_nc", action="store_true",
                   help=f"--case only: delete {NC_FILE} after {RESULT_FILE} and {MEAN_FILE} are written")
    args = p.parse_args()
 
    if args.is_done:
        sys.exit(0 if load_current_result(os.path.abspath(args.is_done)) is not None else 1)
 
    if args.case:
        code, msg = process_and_record(args.case, write_mean=not args.no_mean_file, delete_nc=args.delete_nc)
        print(msg, flush=True)
        sys.exit(code)
 
    dirs = list_case_dirs(args.run_dir, args.case_list)
    if not dirs:
        sys.exit(f"No case folders found (run_dir={args.run_dir!r}, case_list={args.case_list!r})")
 
    if args.write_pending:
        write_pending(dirs, args.write_pending)
        return
    if args.compute_missing:
        compute_missing(dirs, max(1, args.workers), write_mean=not args.no_mean_file)
    collect(dirs, args.out)
 
 
if __name__ == "__main__":
    main()
