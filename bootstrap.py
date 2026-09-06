#!/usr/bin/env python3
"""Install and launch the pinned Qwen GGUF stack on a cloud GPU instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_REPO = "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
MODEL_REVISION = "993a5971fda8f30dd1b7eb2654792ba4415c7460"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"
MODEL_ALIAS = "qwen3.8-27b-aggressive-q5"
MIN_CUDA_VERSION = (12, 8)
MIN_GPU_MEMORY_MIB = 30_000
DISK_SAFETY_BYTES = 15 * 1024**3


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str


MODEL = Asset(
    name="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf",
    size=20_218_177_664,
    sha256="a21e22af885bd2f7c430a72b550a78f5445966e6a72d3143b0950baa0bd6d411",
)
FAST_MTP = Asset(
    name="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf",
    size=903_453_952,
    sha256="115e618e1f73cb50817ed5856f0551c6bf9c3d94df96f440eaca78dc63b8968b",
)
VISION_PROJECTOR = Asset(
    name="mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
    size=931_146_624,
    sha256="5681b690bcb8eb10cd28d62d078cb4e01521a3ea4880a3fc7d54de72de2dd142",
)
FAST_MTP_PATCH = Asset(
    name="HauhauCS-FastMTP-llama.cpp.patch",
    size=2_445,
    sha256="981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615",
)


class SetupError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SetupError(f"{name} must be one of: 1, 0, true, false, yes, no, on, off")


def env_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise SetupError(f"{name} must be an integer, found: {raw_value}") from exc
    if minimum is not None and value < minimum:
        raise SetupError(f"{name} must be at least {minimum}, found: {value}")
    if maximum is not None and value > maximum:
        raise SetupError(f"{name} must be at most {maximum}, found: {value}")
    return value


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    display = " ".join(command)
    log(f"[RUN] {display}")
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def run_retry(
    command: list[str],
    *,
    cwd: Path | None = None,
    attempts: int = 4,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            run(command, cwd=cwd)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            delay = min(5 * attempt, 20)
            log(f"[RETRY] Attempt {attempt}/{attempts} failed; retrying in {delay} seconds")
            time.sleep(delay)


def clone_retry(repository: str, destination: Path, attempts: int = 4) -> None:
    for attempt in range(1, attempts + 1):
        if destination.exists():
            shutil.rmtree(destination)
        try:
            run(["git", "clone", "--filter=blob:none", repository, str(destination)])
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            delay = min(5 * attempt, 20)
            log(f"[RETRY] Git clone attempt {attempt}/{attempts} failed; retrying in {delay} seconds")
            time.sleep(delay)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def asset_is_valid(path: Path, asset: Asset, *, hash_file: bool = True) -> bool:
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    return not hash_file or sha256_file(path).lower() == asset.sha256.lower()


def default_data_dir() -> Path:
    configured = os.environ.get("QWEN_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    workspace = os.environ.get("WORKSPACE")
    if workspace:
        return (Path(workspace) / "qwen-runtime").resolve()
    if Path("/workspace").is_dir():
        return Path("/workspace/qwen-runtime")
    return (Path(__file__).resolve().parent / ".runtime").resolve()


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SetupError(f"Required command was not found: {name}")


def parse_cuda_version() -> tuple[int, int]:
    result = run(["nvcc", "--version"], capture=True)
    match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout or "")
    if not match:
        raise SetupError("Unable to parse the CUDA version from nvcc --version")
    return int(match.group(1)), int(match.group(2))


def inspect_gpu() -> tuple[str, int]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    )
    try:
        first_line = (result.stdout or "").strip().splitlines()[0]
        name, memory_text = [part.strip() for part in first_line.rsplit(",", 1)]
        memory_mib = int(memory_text)
    except (IndexError, ValueError) as exc:
        raise SetupError(f"Unable to parse nvidia-smi output: {result.stdout or ''}") from exc
    return name, memory_mib


def preflight(data_dir: Path, assets: list[Asset]) -> None:
    for command in ("git", "cmake", "ninja", "g++", "nvcc", "nvidia-smi"):
        require_command(command)

    cuda_version = parse_cuda_version()
    if cuda_version < MIN_CUDA_VERSION:
        raise SetupError(
            f"CUDA {cuda_version[0]}.{cuda_version[1]} is too old; CUDA 12.8+ is required"
        )

    gpu_name, memory_mib = inspect_gpu()
    allow_other_gpu = env_bool("QWEN_ALLOW_OTHER_GPU", False)
    if "RTX 5090" not in gpu_name.upper() and not allow_other_gpu:
        raise SetupError(
            f"Expected an RTX 5090, found {gpu_name}. Set QWEN_ALLOW_OTHER_GPU=1 to override."
        )
    if memory_mib < MIN_GPU_MEMORY_MIB:
        raise SetupError(
            f"At least {MIN_GPU_MEMORY_MIB} MiB of VRAM is required, found {memory_mib} MiB"
        )

    missing_bytes = sum(
        asset.size
        for asset in assets
        if not asset_is_valid(data_dir / "models" / asset.name, asset, hash_file=False)
    )
    available_bytes = shutil.disk_usage(data_dir).free
    required_bytes = missing_bytes + DISK_SAFETY_BYTES
    if available_bytes < required_bytes:
        raise SetupError(
            "Insufficient free disk space. "
            f"Required approximately {required_bytes / 1024**3:.1f} GiB, "
            f"available {available_bytes / 1024**3:.1f} GiB."
        )

    log(f"[OK] GPU: {gpu_name} ({memory_mib} MiB)")
    log(f"[OK] CUDA: {cuda_version[0]}.{cuda_version[1]}")
    log(f"[OK] Data directory: {data_dir}")
    log(f"[OK] Free disk space: {available_bytes / 1024**3:.1f} GiB")


def download_asset(asset: Asset, destination_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    destination = destination_dir / asset.name
    if asset_is_valid(destination, asset):
        log(f"[SKIP] Verified asset: {asset.name}")
        return destination

    if destination.exists():
        log(f"[WARN] Removing an invalid asset: {destination}")
        destination.unlink()

    log(f"[DOWNLOAD] {asset.name} ({asset.size / 1_000_000_000:.2f} GB)")
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=asset.name,
                revision=MODEL_REVISION,
                local_dir=destination_dir,
                token=os.environ.get("HF_TOKEN"),
            )
        )
    except Exception as exc:
        raise SetupError(f"Failed to download {asset.name}: {exc}") from exc
    if downloaded.resolve() != destination.resolve():
        shutil.move(str(downloaded), str(destination))
    log(f"[VERIFY] Computing SHA-256 for {asset.name}")
    if not asset_is_valid(destination, asset):
        raise SetupError(f"Downloaded asset failed verification: {asset.name}")
    log(f"[OK] Verified asset: {asset.name}")
    return destination


def prepare_source(data_dir: Path, patch_path: Path) -> tuple[Path, Path]:
    source_dir = data_dir / "source" / f"llama.cpp-{LLAMA_COMMIT[:12]}"
    build_dir = data_dir / "build" / f"llama.cpp-{LLAMA_COMMIT[:12]}-sm120-fastmtp"

    if not (source_dir / ".git").is_dir():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_retry(LLAMA_REPO, source_dir)

    head = run(["git", "rev-parse", "HEAD"], cwd=source_dir, capture=True).stdout.strip()
    if head != LLAMA_COMMIT:
        run_retry(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=source_dir)
        run(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=source_dir)

    forward_check = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if forward_check.returncode == 0:
        run(["git", "apply", str(patch_path)], cwd=source_dir)
        log("[OK] Applied the pinned FastMTP patch")
    else:
        reverse_check = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch_path)],
            cwd=source_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if reverse_check.returncode != 0:
            raise SetupError(
                "The FastMTP patch is neither applicable nor already applied. "
                f"Source directory: {source_dir}"
            )
        log("[SKIP] The pinned FastMTP patch is already applied")

    binary = build_dir / "bin" / "llama-server"
    if not binary.is_file():
        build_dir.mkdir(parents=True, exist_ok=True)
        requested_jobs = env_int("QWEN_BUILD_JOBS", 16, minimum=1)
        jobs = min(os.cpu_count() or 1, requested_jobs)
        run(
            [
                "cmake",
                "-S",
                str(source_dir),
                "-B",
                str(build_dir),
                "-G",
                "Ninja",
                "-DGGML_CUDA=ON",
                "-DGGML_NATIVE=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=120",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_CURL=OFF",
            ]
        )
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "llama-server",
                "-j",
                str(jobs),
            ]
        )
    else:
        log(f"[SKIP] Existing llama-server build: {binary}")

    if not binary.is_file():
        raise SetupError(f"llama-server was not produced at the expected path: {binary}")
    return source_dir, binary


def create_api_key(secret_dir: Path) -> tuple[str, Path]:
    secret_dir.mkdir(parents=True, exist_ok=True)
    key_path = secret_dir / "api-key.txt"
    configured = os.environ.get("QWEN_API_KEY", "").strip()
    if configured:
        api_key = configured
        key_path.write_text(api_key + "\n", encoding="utf-8")
    elif key_path.is_file() and key_path.read_text(encoding="utf-8").strip():
        api_key = key_path.read_text(encoding="utf-8").strip()
    else:
        api_key = secrets.token_urlsafe(36)
        key_path.write_text(api_key + "\n", encoding="utf-8")
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return api_key, key_path


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_existing(pid_path: Path) -> None:
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return
    if not pid_is_running(pid):
        pid_path.unlink(missing_ok=True)
        return
    log(f"[STOP] Stopping existing llama-server process {pid}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not pid_is_running(pid):
            pid_path.unlink(missing_ok=True)
            return
        time.sleep(1)
    raise SetupError(f"Existing llama-server process {pid} did not stop")


def server_command(
    binary: Path,
    model_path: Path,
    draft_path: Path | None,
    projector_path: Path | None,
    key_path: Path,
    host: str,
    port: int,
    context_size: int,
    mtp_mode: str,
    flash_attention: bool,
) -> list[str]:
    command = [
        str(binary),
        "--model",
        str(model_path),
        "--alias",
        MODEL_ALIAS,
        "--ctx-size",
        str(context_size),
        "--parallel",
        "1",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--flash-attn",
        "on" if flash_attention else "off",
        "--temp",
        "1.0",
        "--top-k",
        "20",
        "--top-p",
        "0.95",
        "--min-p",
        "0",
        "--presence-penalty",
        "0",
        "--repeat-penalty",
        "1.0",
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-effort",
        "xhigh",
        "--reasoning-preserve",
        "--reasoning-format",
        "deepseek",
        "--host",
        host,
        "--port",
        str(port),
        "--api-key-file",
        str(key_path),
    ]

    if projector_path is not None:
        command.extend(["--mmproj", str(projector_path)])

    command.extend(["--spec-type", "draft-mtp"])
    if mtp_mode == "fast":
        if draft_path is None:
            raise SetupError("FastMTP mode requires the draft model")
        command.extend(
            [
                "--spec-draft-model",
                str(draft_path),
                "--spec-draft-ngl",
                "all",
                "--spec-draft-n-max",
                "3",
                "--spec-draft-p-min",
                "0",
            ]
        )
    else:
        command.extend(
            [
                "--spec-draft-ngl",
                "all",
                "--spec-draft-n-max",
                "2",
            ]
        )
    return command


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict | None = None,
    timeout: int = 10,
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body}
        return exc.code, parsed


def log_tail(log_path: Path, lines: int = 80) -> str:
    if not log_path.is_file():
        return "Log file was not created."
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def wait_until_ready(process: subprocess.Popen[bytes], port: int, log_path: Path) -> None:
    timeout_seconds = env_int("QWEN_START_TIMEOUT", 600, minimum=30)
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise SetupError(
                f"llama-server exited with code {return_code}.\n\n{log_tail(log_path)}"
            )
        try:
            status, _ = request_json(health_url, timeout=5)
            if status == 200:
                log("[OK] llama-server health check passed")
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(2)
    process.terminate()
    raise SetupError(f"llama-server did not become ready in time.\n\n{log_tail(log_path)}")


def smoke_test(port: int, api_key: str) -> None:
    payload = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "Reply with the single word READY."}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, body = request_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout=180,
    )
    choices = body.get("choices") if isinstance(body, dict) else None
    if status != 200 or not choices:
        raise SetupError(f"Inference smoke test failed with HTTP {status}: {body}")
    log("[OK] Inference smoke test passed")


def public_url(port: int) -> str | None:
    runpod_id = os.environ.get("RUNPOD_POD_ID")
    if runpod_id:
        return f"https://{runpod_id}-{port}.proxy.runpod.net"

    vast_port = os.environ.get(f"VAST_TCP_PORT_{port}")
    vast_ip = os.environ.get("PUBLIC_IPADDR") or os.environ.get("VAST_PUBLIC_IP")
    if vast_port and vast_ip:
        return f"http://{vast_ip}:{vast_port}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the server managed by this bootstrap and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = default_data_dir()
    model_dir = data_dir / "models"
    artifact_dir = data_dir / "artifacts"
    log_dir = data_dir / "logs"
    run_dir = data_dir / "run"
    secret_dir = data_dir / "secrets"
    for directory in (data_dir, model_dir, artifact_dir, log_dir, run_dir, secret_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pid_path = run_dir / "llama-server.pid"
    if args.stop:
        stop_existing(pid_path)
        log("[OK] llama-server is stopped")
        return 0

    mtp_mode = os.environ.get("QWEN_MTP_MODE", "fast").strip().lower()
    if mtp_mode not in {"fast", "embedded"}:
        raise SetupError("QWEN_MTP_MODE must be either fast or embedded")
    enable_vision = env_bool("QWEN_ENABLE_VISION", True)
    flash_attention = env_bool("QWEN_FLASH_ATTN", True)
    context_size = env_int("QWEN_CTX_SIZE", 32_768, minimum=1_024, maximum=262_144)
    port = env_int("QWEN_PORT", 8_000, minimum=1, maximum=65_535)
    host = os.environ.get("QWEN_HOST", "0.0.0.0")

    required_assets = [MODEL]
    if mtp_mode == "fast":
        required_assets.append(FAST_MTP)
    if enable_vision:
        required_assets.append(VISION_PROJECTOR)
    preflight(data_dir, required_assets)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    os.environ.setdefault("HF_HOME", str(data_dir / ".hf-home"))

    patch_path = download_asset(FAST_MTP_PATCH, artifact_dir)
    _, binary = prepare_source(data_dir, patch_path)
    model_path = download_asset(MODEL, model_dir)
    draft_path = download_asset(FAST_MTP, model_dir) if mtp_mode == "fast" else None
    projector_path = download_asset(VISION_PROJECTOR, model_dir) if enable_vision else None

    api_key, key_path = create_api_key(secret_dir)
    stop_existing(pid_path)
    command = server_command(
        binary,
        model_path,
        draft_path,
        projector_path,
        key_path,
        host,
        port,
        context_size,
        mtp_mode,
        flash_attention,
    )
    log_path = log_dir / "llama-server.log"
    log(f"[START] Launching llama-server; log: {log_path}")
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")

    try:
        wait_until_ready(process, port, log_path)
        smoke_test(port, api_key)
    except Exception:
        if process.poll() is None:
            process.terminate()
        pid_path.unlink(missing_ok=True)
        raise

    url = public_url(port)
    log("")
    log("Setup completed successfully.")
    log(f"Web UI (inside the instance): http://127.0.0.1:{port}")
    if url:
        log(f"Public URL: {url}")
    else:
        log("Public URL: use the port mapping shown by your cloud platform")
    log(f"OpenAI base URL: {(url or f'http://127.0.0.1:{port}')}/v1")
    log(f"Model name: {MODEL_ALIAS}")
    log(f"API key: {api_key}")
    log(f"API key file: {key_path}")
    log(f"Log file: {log_path}")
    log(f"PID file: {pid_path}")
    log(f"Stop command: bash {Path(__file__).resolve().parent / 'bootstrap.sh'} --stop")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(exc.returncode or 1)
