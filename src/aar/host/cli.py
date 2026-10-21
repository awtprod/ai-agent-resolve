"""``aar host`` commands: verified transfers and PowerShell on the edit PC.

``pull`` and ``push`` move a file and check its size and SHA-256; ``hash`` reports a remote
file's size and SHA-256; ``ps`` runs a PowerShell script *file* (never an inline string) on the
edit PC. The host comes from ``AAR_EDIT_*`` (see :mod:`aar.host.config`); there is no default.
Exit status: 0 ok, 1 the transfer or remote command failed, 2 bad input or configuration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from aar import cli_io
from aar.host.config import MAX_TIMEOUT_SECONDS, check_timeout, env_config
from aar.host.errors import ConfigError, HostError, PathRefused, ScriptTooLong, TransportError
from aar.host.runner import Runner, SubprocessRunner
from aar.host.transport import EditHost, TransferResult

_PROG = "aar host"
EXIT_OK = 0
EXIT_FAILED = 1  # the transfer or remote command failed
EXIT_INPUT = 2  # bad arguments, path refused, or unusable configuration


def register(subparsers: Any) -> None:
    """Add the ``host`` command group to the top-level parser."""
    host = subparsers.add_parser(
        "host",
        help="Edit-PC transport: verified scp transfers and PowerShell.",
        description=(
            "Move files to and from the edit PC with size and SHA-256 checks, and run PowerShell "
            "there. The host is named only by AAR_EDIT_* configuration. Exit: 0 ok, 1 failed, "
            "2 bad input or configuration."
        ),
    )
    sub = host.add_subparsers(dest="host_command", metavar="COMMAND", required=True)

    pull = sub.add_parser("pull", help="Download a remote file to a local path, verified.")
    pull.add_argument("remote", help="Edit-PC (Windows) source path.")
    pull.add_argument("local", help="Hub (local) destination path.")
    pull.add_argument("--overwrite", action="store_true", help="Replace the local file if present.")
    pull.add_argument("--json", action="store_true", help="Print the result as JSON.")
    pull.set_defaults(func=_cmd_pull)

    push = sub.add_parser("push", help="Upload a local file to a remote path, verified.")
    push.add_argument("local", help="Hub (local) source path.")
    push.add_argument("remote", help="Edit-PC (Windows) destination path.")
    push.add_argument(
        "--overwrite", action="store_true", help="Replace the remote file if present."
    )
    push.add_argument("--json", action="store_true", help="Print the result as JSON.")
    push.set_defaults(func=_cmd_push)

    hash_ = sub.add_parser("hash", help="Report a remote file's size and SHA-256.")
    hash_.add_argument("remote", help="Edit-PC (Windows) path.")
    hash_.add_argument("--json", action="store_true", help="Print the result as JSON.")
    hash_.set_defaults(func=_cmd_hash)

    ps = sub.add_parser("ps", help="Run a PowerShell script file on the edit PC.")
    ps.add_argument("script", help="Path to a local PowerShell script file (not an inline string).")
    ps.add_argument(
        "--timeout",
        type=_timeout_arg,
        help=f"Override the command timeout: finite seconds in (0, {MAX_TIMEOUT_SECONDS}].",
    )
    ps.add_argument("--json", action="store_true", help="Print the result as JSON.")
    ps.set_defaults(func=_cmd_ps)


def _timeout_arg(raw: str) -> float:
    """An argparse type for --timeout: a number of seconds in (0, MAX_TIMEOUT_SECONDS].

    Shares :func:`check_timeout` with config and the library override (NF3/NF6), so nan, inf, 0,
    negatives and over-large values exit 2 with a usage error rather than reaching subprocess.
    """
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {raw!r}") from None
    try:
        return check_timeout(value, "timeout")
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _default_runner() -> Runner:
    """The runner the CLI uses (overridden in tests)."""
    return SubprocessRunner()


def _host() -> EditHost:
    return EditHost(env_config(), _default_runner())


def _error(exc: Exception, code: int, as_json: bool) -> int:
    kind = getattr(exc, "kind", type(exc).__name__)
    if as_json:
        error: dict[str, object] = {"type": kind, "message": str(exc)}
        if isinstance(exc, PathRefused):
            error["reason"] = exc.reason
        body = {"ok": False, "error": error}
        return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or code
    cli_io.warn(f"error: {exc}")
    return code


def _emit(result: TransferResult, as_json: bool, human: str) -> int:
    text = json.dumps(result.to_dict(), indent=2) + "\n" if as_json else human + "\n"
    return cli_io.emit(_PROG, text) or EXIT_OK


def _dispatch(as_json: bool, run: Callable[[], int]) -> int:
    try:
        return run()
    except (ConfigError, PathRefused, ScriptTooLong) as exc:
        return _error(exc, EXIT_INPUT, as_json)
    except TransportError as exc:
        return _error(exc, EXIT_FAILED, as_json)
    except HostError as exc:  # any other typed host error
        return _error(exc, EXIT_FAILED, as_json)
    except OSError as exc:
        return _error(exc, EXIT_FAILED, as_json)


def _cmd_pull(args: argparse.Namespace) -> int:
    def run() -> int:
        result = _host().pull(args.remote, args.local, overwrite=args.overwrite)
        return _emit(
            result, args.json, f"pulled {result.size} bytes  {result.sha256}  -> {result.local}"
        )

    return _dispatch(args.json, run)


def _cmd_push(args: argparse.Namespace) -> int:
    def run() -> int:
        result = _host().push(args.local, args.remote, overwrite=args.overwrite)
        return _emit(
            result, args.json, f"pushed {result.size} bytes  {result.sha256}  -> {result.remote}"
        )

    return _dispatch(args.json, run)


def _cmd_hash(args: argparse.Namespace) -> int:
    def run() -> int:
        result = _host().file_hash(args.remote)
        return _emit(result, args.json, f"{result.sha256}  {result.size}  {result.remote}")

    return _dispatch(args.json, run)


def _cmd_ps(args: argparse.Namespace) -> int:
    def run() -> int:
        try:
            script = _read_text(args.script)
        except OSError as exc:
            return _error(exc, EXIT_INPUT, args.json)
        except UnicodeDecodeError as exc:
            # a .ps1 saved as UTF-16 (Windows ISE default) or otherwise not UTF-8: a typed input
            # error with exit 2, never a traceback.
            return _error(
                ValueError(f"{args.script}: not a UTF-8 PowerShell script ({exc})"),
                EXIT_INPUT,
                args.json,
            )
        try:
            stdout = _host().run_powershell(script, timeout=args.timeout)
        except ValueError as exc:
            # a NUL in the script (e.g. a BOM-less UTF-16 file decoded as UTF-8): typed, exit 2.
            return _error(ValueError(f"{args.script}: {exc}"), EXIT_INPUT, args.json)
        if args.json:
            body = {"ok": True, "op": "ps", "stdout": stdout}
            return cli_io.emit(_PROG, json.dumps(body, indent=2) + "\n") or EXIT_OK
        return cli_io.emit(_PROG, stdout) or EXIT_OK

    return _dispatch(args.json, run)


def _read_text(path: str) -> str:
    # utf-8-sig: strip a UTF-8 BOM so it is not sent as a leading U+FEFF. A non-UTF-8 file raises
    # UnicodeDecodeError (handled in _cmd_ps as a typed exit-2 refusal).
    with open(path, encoding="utf-8-sig") as handle:
        return handle.read()
