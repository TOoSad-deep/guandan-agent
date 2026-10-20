"""Explicit config checks, local game runs and one-worker web serving."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import tomllib

from pydantic import ValidationError

from guandan_agent import __version__


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite positive number") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="掼蛋 Agent 工程与配置检查")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("config-check", help="校验并展示非敏感配置，不启动对局或模型")
    check.add_argument("path", type=Path)
    check.add_argument("--model", help="显式覆盖 model.name；不读取环境变量或凭证")
    run = commands.add_parser("run", help="按指定配置启动本地平台对局")
    run.add_argument("path", type=Path)
    run.add_argument("--recording-root", type=Path, default=Path("runs"))
    run.add_argument("--startup-timeout-s", type=_positive_seconds, default=15.0)
    run.add_argument("--max-runtime-s", type=_positive_seconds,
                     help="显式开发运行时长，到时停止；不会伪造比赛结束")
    serve = commands.add_parser("serve", help="启动本地单进程牌桌与只读历史查询")
    serve.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--recording-root", type=Path, default=Path("runs"))
    serve.add_argument("--archive-root", type=Path)
    serve.add_argument("--config-root", type=Path, default=Path("configs"))
    serve.add_argument("--allow-config", action="append", dest="allowed_configs",
                       help="允许页面启动的 TOML 文件基名；默认仅 local.example.toml")
    review = commands.add_parser("review", help="检查已关闭源；显式 --live 才生成离线复盘")
    review.add_argument("path", type=Path, help="本批离线调用配置")
    review.add_argument("sources", nargs="+", type=Path)
    review.add_argument("--live", action="store_true")
    review.add_argument("--session-id")
    review.add_argument("--request-id", action="append", dest="request_ids")
    evaluate = commands.add_parser("evaluate", help="离线重算记录统计，不调用模型")
    evaluate.add_argument("sources", nargs="+", type=Path)
    evaluate.add_argument("--unit", choices=("match", "round"), default="match")
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        from guandan_agent.app import evaluate_archives
        try:
            result = asyncio.run(evaluate_archives(tuple(args.sources), output=args.output, unit=args.unit))
        except KeyboardInterrupt:
            return 130
        except Exception:
            print('{"error":"EVALUATION_FAILED"}', file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        from guandan_agent.app import create_web_application
        import uvicorn
        if not 1 <= args.port <= 65535:
            print('{"error":"INVALID_WEB_PORT"}', file=sys.stderr)
            return 2
        try:
            application = create_web_application(
                recording_root=args.recording_root, config_root=args.config_root,
                allowed_config_names=tuple(args.allowed_configs or ("local.example.toml",)),
                archive_root=args.archive_root, host=args.host, port=args.port,
            )
            uvicorn.run(application, host=args.host, port=args.port, workers=1, reload=False,
                        access_log=False)
        except KeyboardInterrupt:
            return 130
        except Exception:
            print('{"error":"WEB_UNAVAILABLE"}', file=sys.stderr)
            return 2
        return 0

    from guandan_agent.config import load_config

    overrides = ({"model": {"name": args.model}}
                 if getattr(args, "model", None) is not None else None)
    try:
        config = load_config(args.path, overrides=overrides)
    except ValidationError as exc:
        # Do not echo input values: a user may accidentally paste a secret into TOML.
        errors = [{"field": ".".join(map(str, e["loc"])), "type": e["type"]}
                  for e in exc.errors(include_input=False, include_context=False)]
        print(json.dumps({"error": "invalid_config", "details": errors}, ensure_ascii=False),
              file=sys.stderr)
        return 2
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        print("无法读取有效的 UTF-8 TOML 配置文件。", file=sys.stderr)
        return 2
    if args.command == "review":
        from guandan_agent.app import review_archives
        try:
            result = asyncio.run(review_archives(config, tuple(args.sources), live=args.live,
                                 session_id=args.session_id,
                                 request_ids=None if args.request_ids is None else tuple(args.request_ids)))
        except KeyboardInterrupt:
            print('{"error":"REVIEW_INTERRUPTED"}', file=sys.stderr)
            return 130
        except Exception:
            print('{"error":"REVIEW_FAILED"}', file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not args.live or any(item["report"]["items"] for item in result["reports"]) else 1
    if args.command == "run":
        from guandan_agent import app
        from guandan_agent.runtime.manager import RunManagerError

        try:
            result = asyncio.run(app.run_config(
                config, recording_root=args.recording_root,
                startup_timeout_s=args.startup_timeout_s,
                max_runtime_s=args.max_runtime_s,
            ))
        except KeyboardInterrupt:
            print("运行已停止。", file=sys.stderr)
            return 130
        except RunManagerError as exc:
            print(json.dumps({"error": exc.error.model_dump(mode="json")}, ensure_ascii=False),
                  file=sys.stderr)
            return 2
        except Exception:
            # Exceptions at the process boundary must not reveal URLs, config
            # values or transport contents. Runtime details belong to safe logs.
            print('{"error":"RUN_FAILED"}', file=sys.stderr)
            return 1
        print(result.model_dump_json(indent=2))
        return 0 if result.status == "completed" else 1
    print(json.dumps({"config_id": config.config_id,
                      "config": config.model_dump(mode="json")}, ensure_ascii=False, indent=2))
    return 0
