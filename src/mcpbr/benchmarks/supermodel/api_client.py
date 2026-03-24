"""Async Supermodel API client with polling and idempotency support."""

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time

logger = logging.getLogger("mcpbr.supermodel")


async def call_supermodel_api(
    endpoint_path: str,
    zip_path: str,
    api_base: str,
    api_key: str | None = None,
    idempotency_key: str | None = None,
    max_poll_time: int = 600,
) -> dict:
    """Call a Supermodel API endpoint with a zipped repo.

    Uses curl subprocess for the HTTP request and polls for async results.

    Args:
        endpoint_path: API endpoint path (e.g. '/v1/analysis/dead-code').
        zip_path: Path to the zipped repository archive.
        api_base: Base URL for the Supermodel API.
        api_key: Optional API key.
        idempotency_key: Optional idempotency key (auto-generated from zip hash if not provided).
        max_poll_time: Maximum time to poll for results in seconds.

    Returns:
        Parsed API response dict.

    Raises:
        RuntimeError: If the API request fails or times out.
    """
    url = f"{api_base}{endpoint_path}"

    if not idempotency_key:
        with open(zip_path, "rb") as f:
            zip_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        ep_name = endpoint_path.strip("/").replace("/", "-")
        idempotency_key = f"bench:{ep_name}:{zip_hash}:v2"

    headers = [
        "-H",
        "Accept: application/json",
        "-H",
        f"Idempotency-Key: {idempotency_key}",
    ]

    # Pass API key via curl config file to avoid exposure in process table (ps aux)
    api_key_config_path: str | None = None
    if api_key:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cfg", prefix="mcpbr_curl_", delete=False
        ) as api_key_fd:
            api_key_fd.write(f'header = "X-Api-Key: {api_key}"\n')
            api_key_config_path = api_key_fd.name
        os.chmod(api_key_config_path, 0o600)

    # Initial request with file upload
    upload_cmd = ["curl", "-s", "-X", "POST", url, "-F", f"file=@{zip_path}", *headers]
    if api_key_config_path:
        upload_cmd.extend(["--config", api_key_config_path])

    start_time = time.time()
    print(
        f"  Supermodel API: uploading {zip_path} to {endpoint_path}...", file=sys.stderr, flush=True
    )

    proc = await asyncio.create_subprocess_exec(
        *upload_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)

    if proc.returncode != 0:
        raise RuntimeError(f"Supermodel API request failed: {stderr.decode()}")

    try:
        response = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON response from Supermodel API: {stdout.decode()[:500]}") from e

    # Poll if async — use lightweight requests (1-byte dummy file instead of
    # re-uploading the full zip). The API recognizes the idempotency key and
    # returns the cached job status without reprocessing.
    poll_dummy_path: str | None = None
    poll_count = 0

    try:
        while response.get("status") in ("pending", "processing"):
            elapsed = time.time() - start_time
            if elapsed > max_poll_time:
                raise RuntimeError(f"Supermodel API timed out after {max_poll_time}s")

            # Create poll dummy on first iteration only
            if poll_dummy_path is None:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as poll_dummy:
                    poll_dummy.write(b"\n")
                    poll_dummy_path = poll_dummy.name

            poll_cmd = [
                "curl",
                "-s",
                "-X",
                "POST",
                url,
                "-F",
                f"file=@{poll_dummy_path}",
                *headers,
            ]
            if api_key_config_path:
                poll_cmd.extend(["--config", api_key_config_path])

            retry_after = response.get("retryAfter", 10)
            poll_count += 1
            print(
                f"  Supermodel API: {response.get('status')} "
                f"(poll #{poll_count}, {elapsed:.0f}s elapsed, retry in {retry_after}s)",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(retry_after)

            proc = await asyncio.create_subprocess_exec(
                *poll_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                raise RuntimeError(f"Supermodel API poll failed: {stderr.decode()}")
            try:
                response = json.loads(stdout.decode())
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Non-JSON poll response from Supermodel API: {stdout.decode()[:500]}"
                ) from e
    finally:
        if poll_dummy_path is not None:
            os.unlink(poll_dummy_path)
        if api_key_config_path is not None:
            os.unlink(api_key_config_path)

    elapsed = time.time() - start_time

    # Check for error responses (status can be string "error" or HTTP status int)
    status = response.get("status")
    if status == "error" or response.get("error"):
        raise RuntimeError(
            f"Supermodel API error: {response.get('error', response.get('message'))}"
        )
    if isinstance(status, int) and status >= 400:
        raise RuntimeError(f"Supermodel API HTTP {status}: {response.get('message', response)}")

    api_result = response.get("result", response)
    print(f"  Supermodel API: completed in {elapsed:.1f}s", file=sys.stderr, flush=True)
    return dict(api_result)
