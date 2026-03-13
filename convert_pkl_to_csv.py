"""
Convert a (potentially corrupted) pickle DataFrame file to CSV.

Strategy:
  1. Try pandas.read_pickle (fastest, handles pandas-specific metadata)
  2. Fall back to raw pickle.load
  3. Fall back to pickle with latin-1 encoding (handles some encoding corruption)
  4. Fall back to partial recovery using pickletools scan + custom Unpickler
"""

import csv
import io
import os
import pickle
import pickletools
import struct
import sys


PKL_FILE = "full_network_mobility_scores.pkl"
CSV_FILE = "full_network_mobility_scores.csv"


# ---------------------------------------------------------------------------
# Helper: save whatever we have to CSV
# ---------------------------------------------------------------------------

def save_to_csv(obj, csv_path: str) -> int:
    """Write obj (DataFrame, list-of-dicts, list-of-lists, or dict) to CSV.

    Returns the number of rows written.
    """
    # pandas DataFrame
    try:
        import pandas as pd  # noqa: PLC0415
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(csv_path, index=False)
            return len(obj)
    except ImportError:
        pass

    rows_written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if isinstance(obj, dict):
            # dict-of-columns  ->  transpose into rows
            writer = csv.DictWriter(f, fieldnames=list(obj.keys()))
            writer.writeheader()
            length = len(next(iter(obj.values())))
            for i in range(length):
                writer.writerow({k: v[i] for k, v in obj.items()})
                rows_written += 1
        elif isinstance(obj, (list, tuple)) and obj:
            first = obj[0]
            if isinstance(first, dict):
                writer = csv.DictWriter(f, fieldnames=list(first.keys()))
                writer.writeheader()
                for row in obj:
                    writer.writerow(row)
                    rows_written += 1
            else:
                writer = csv.writer(f)
                for row in obj:
                    writer.writerow(row if isinstance(row, (list, tuple)) else [row])
                    rows_written += 1
        else:
            # Last resort: write repr
            f.write(repr(obj))
            rows_written = 1

    return rows_written


# ---------------------------------------------------------------------------
# Attempt 1 – pandas.read_pickle
# ---------------------------------------------------------------------------

def try_pandas(pkl_path: str):
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_pickle(pkl_path)
        print("[attempt 1] pandas.read_pickle succeeded.")
        return df
    except ImportError:
        print("[attempt 1] pandas not installed, skipping.")
    except Exception as exc:
        print(f"[attempt 1] pandas.read_pickle failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Attempt 2 – raw pickle.load (default encoding)
# ---------------------------------------------------------------------------

def try_raw_pickle(pkl_path: str, encoding: str = "ASCII"):
    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        print(f"[attempt 2] pickle.load (encoding={encoding}) succeeded.")
        return obj
    except Exception as exc:
        print(f"[attempt 2] pickle.load failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Attempt 3 – latin-1 encoding (rescues files pickled on Python 2)
# ---------------------------------------------------------------------------

def try_pickle_latin1(pkl_path: str):
    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh, encoding="latin-1")  # noqa: S301
        print("[attempt 3] pickle.load(encoding='latin-1') succeeded.")
        return obj
    except Exception as exc:
        print(f"[attempt 3] pickle.load(encoding='latin-1') failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Attempt 4 – partial recovery: scan for string / numeric pairs
# ---------------------------------------------------------------------------

class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that substitutes unknown globals with a placeholder."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            print(f"  [safe unpickler] unknown global {module}.{name} – using placeholder")
            return lambda *a, **kw: None


def try_safe_unpickle(pkl_path: str):
    try:
        with open(pkl_path, "rb") as fh:
            obj = _SafeUnpickler(fh).load()  # noqa: S301
        print("[attempt 4] SafeUnpickler succeeded.")
        return obj
    except Exception as exc:
        print(f"[attempt 4] SafeUnpickler failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# Attempt 5 – brute-force scan for float32 / float64 runs
# ---------------------------------------------------------------------------

def try_brute_force_floats(pkl_path: str):
    """Scan binary file for IEEE-754 float runs as a last resort."""
    MIN_RUN = 1000  # minimum consecutive valid floats to consider a block

    print("[attempt 5] Brute-force scanning for float64 runs …")
    results = []
    with open(pkl_path, "rb") as fh:
        data = fh.read()

    # Scan every 8-byte-aligned offset for float64 values in a sane range
    for offset in range(0, len(data) - 8, 8):
        try:
            val = struct.unpack_from("<d", data, offset)[0]
            if not (-1e15 < val < 1e15):
                continue
        except struct.error:
            continue
        # Start of a run
        run = [val]
        pos = offset + 8
        while pos + 8 <= len(data):
            try:
                v = struct.unpack_from("<d", data, pos)[0]
            except struct.error:
                break
            if not (-1e15 < v < 1e15):
                break
            run.append(v)
            pos += 8
        if len(run) >= MIN_RUN:
            results.append(run)
            print(f"  found run of {len(run)} float64 values at offset {offset}")

    if not results:
        print("[attempt 5] No usable float runs found.")
        return None

    # Flatten all runs into a single column DataFrame / list
    all_vals = [v for run in results for v in run]
    try:
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame({"value": all_vals})
    except ImportError:
        return [{"value": v} for v in all_vals]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(PKL_FILE):
        print(f"ERROR: '{PKL_FILE}' not found in the current directory.")
        sys.exit(1)

    print(f"Converting '{PKL_FILE}' → '{CSV_FILE}' …\n")

    obj = None
    for attempt in (try_pandas, try_raw_pickle, try_pickle_latin1, try_safe_unpickle):
        obj = attempt(PKL_FILE)
        if obj is not None:
            break

    if obj is None:
        obj = try_brute_force_floats(PKL_FILE)

    if obj is None:
        print("\nAll recovery attempts failed. The file may be too severely corrupted.")
        sys.exit(1)

    print(f"\nObject type  : {type(obj).__name__}")
    try:
        print(f"Shape        : {obj.shape}")
        print(f"Columns      : {list(obj.columns)}")
        print(f"Preview:\n{obj.head()}")
    except AttributeError:
        pass

    rows = save_to_csv(obj, CSV_FILE)
    print(f"\nSaved {rows:,} rows to '{CSV_FILE}'.")


if __name__ == "__main__":
    main()
