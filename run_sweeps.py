import json
import subprocess
import sys
from typing import Any, Dict, List


def _bool_flag(v: Any) -> str:
    return "true" if bool(v) else "false"


def build_cmd(cfg: Dict[str, Any], data_name: str, sweep: Dict[str, Any], nm: float) -> List[str]:
    py = cfg.get("python", sys.executable)
    entry = cfg.get("entrypoint", "main.py")

    global_params = cfg.get("global", {})
    ds_params = cfg.get("datasets", {}).get(data_name, {})
    sweep_params = sweep.get("params", {})

    method = sweep["method"]
    base_outdir = cfg.get("base_outdir", "./runs")

    args: List[str] = [
        py,
        entry,
        f"--dir={base_outdir}",
        f"--data={data_name}",
        f"--method={method}",
        f"--noise_multiplier={nm}",
    ]

    # dataset-level required args
    if "batch_size" in ds_params:
        args.append(f"--batch_size={ds_params['batch_size']}")

    # global args (can be overridden by sweep params)
    for k, v in global_params.items():
        if k in sweep_params:
            continue
        if isinstance(v, bool):
            args.append(f"--{k}={_bool_flag(v)}")
        else:
            args.append(f"--{k}={v}")

    # sweep overrides / extra params
    for k, v in sweep_params.items():
        if isinstance(v, bool):
            args.append(f"--{k}={_bool_flag(v)}")
        else:
            args.append(f"--{k}={v}")

    # random batch delta only needed when random_batch=true
    rb = bool(sweep_params.get("random_batch", False))
    if rb:
        if "random_batch_delta" in ds_params:
            args.append(f"--random_batch_delta={ds_params['random_batch_delta']}")
        else:
            args.append("--random_batch_delta=1e-5")

    # guardrail: dp_sgd_amp requires random_batch=false
    if method == "dp_sgd_amp" and rb:
        raise ValueError(
            "dp_sgd_amp requires random_batch=false (amplification assumes uniform subsampling)."
        )

    return args


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    sweeps = cfg.get("sweeps", [])

    # Number of repetitions per setting (optional). If not provided, defaults to 1.
    # This is *not* seed control; it's just running the same command multiple times.
    total = 0
    for sweep in sweeps:
        repeats = int(sweep.get("repeats", 1))
        total += len(sweep["data"]) * len(sweep["noise_multiplier"]) * repeats

    print(f"Planned runs: {total}")
    run_idx = 0

    for sweep in sweeps:
        sweep_name = sweep.get("name", sweep.get("method", "sweep"))
        print(f"\n=== Sweep: {sweep_name} (method={sweep['method']}) ===")
        for data_name in sweep["data"]:
            for nm in sweep["noise_multiplier"]:
                repeats = int(sweep.get("repeats", 1))
                if "seeds" in sweep:
                    print("[WARN] 'seeds' is present in sweep_config.json but is ignored. "
                          "This runner no longer sets seeds; use 'repeats' for multiple trials.")
                for rep in range(repeats):
                    run_idx += 1
                    cmd = build_cmd(cfg, data_name, sweep, nm)
                    suffix = f" (rep {rep + 1}/{repeats})" if repeats > 1 else ""
                    print(f"\n[{run_idx}/{total}]{suffix} " + " ".join(cmd))
                    subprocess.run(cmd, check=True)

    print("\nAll sweeps completed.")


if __name__ == "__main__":
    main()
