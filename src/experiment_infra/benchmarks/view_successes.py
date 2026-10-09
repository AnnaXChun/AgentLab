"""Build an offline, success-only trajectory reader from canonical benchmark exports."""

import argparse
import html
import json
import os
from pathlib import Path


def block(title, value, opened=False):
    body = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
    return (
        f"<details{' open' if opened else ''}><summary>{html.escape(title)}</summary>"
        f"<pre>{html.escape(body)}</pre></details>"
    )


def page(title, body):
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'
        + html.escape(title)
        + """</title>
<style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#18222d;background:#f7f9fc}
a{color:#1768aa}h1{font-size:28px}details{background:white;border:1px solid #d8e0ea;border-radius:8px;padding:12px;margin:12px 0}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace}
li{margin:12px 0}.note{color:#536475}</style>"""
        + body
        + "</html>"
    )


def build_reader(directory):
    root = Path(directory).resolve()
    destination = root / "successes"
    destination.mkdir(exist_ok=True)
    links, index = [], []
    for journal in sorted((root / "tasks").glob("*.json")):
        row = json.loads(journal.read_text())
        if (
            row.get("status") != "complete"
            or not row.get("success")
            or not row.get("export_directory")
        ):
            continue
        bundle = Path(row["export_directory"])
        trajectories = [
            json.loads(s) for s in (bundle / "trajectory.jsonl").read_text().splitlines()
        ]
        original = next(t for t in trajectories if t["trajectory_id"] == row["trajectory_id"])
        artifacts = {
            a["artifact_id"]: a
            for a in map(json.loads, (bundle / "artifacts.jsonl").read_text().splitlines())
        }
        title = f"{row['benchmark']} · {row['task_id']}"
        body = "<h1>" + html.escape(title) + '</h1><p><a href="index.html">返回成功列表</a></p>'
        body += block(
            "评测与运行信息",
            {
                k: row.get(k)
                for k in [
                    "run_id",
                    "trajectory_id",
                    "evidence_id",
                    "claim_id",
                    "model_name",
                    "temperature",
                    "seed",
                    "steps",
                    "total_tokens",
                    "termination",
                    "failure",
                    "native_result",
                    "replay",
                ]
            },
            True,
        )
        bundle_link = html.escape(os.path.relpath(bundle, destination), quote=True)
        body += f'<p><a href="{bundle_link}/trajectory.jsonl">原始完整轨迹</a> · <a href="{bundle_link}/manifest.json">导出清单与哈希</a> · <a href="{bundle_link}/evidence.jsonl">官方评测 Evidence</a></p>'
        body += "<h2>模型请求（含完整 system/user prompt 和历史）</h2>"
        for i, aid in enumerate(row.get("request_refs", []), 1):
            digest = artifacts[aid]["sha256"]
            request = json.loads((bundle / "artifacts/sha256" / digest[:2] / digest).read_text())
            body += block(f"模型调用 {i}：完整请求", request)
        body += "<h2>模型返回（含结束决策、token 用量和诊断）</h2>"
        for i, call in enumerate(row.get("model_calls", []), 1):
            body += block(f"模型调用 {i}：完整返回记录", call)
        body += "<h2>完整执行步骤</h2>"
        for i, step in enumerate(original["steps"], 1):
            action = step["action"]
            command = action.get("arguments", {}).get("command", {}).get("argv", [])
            label = command[1] if len(command) > 1 and command[0] == "tau2" else action["operation"]
            if action["tool"] == "mcp" and action.get("arguments", {}).get("tool_name") == (
                "benchmark.native_evaluate"
            ):
                label = "官方评测（由驱动发起）"
            body += f"<h3>步骤 {i} · {html.escape(label)}</h3>"
            body += block("动作", action, True)
            body += block("工具返回", step["observation"], True)
            body += block("完整步骤（含前后状态、快照引用和溯源）", step)
        name = row["run_id"] + ".html"
        (destination / name).write_text(page(title, body))
        links.append(
            f'<li><a href="{name}">{html.escape(title)}</a> — {row["steps"]} 个模型动作，官方得分 {row["native_result"]["score"]}</li>'
        )
        index.append(
            {
                "task_id": row["task_id"],
                "run_id": row["run_id"],
                "trajectory_id": row["trajectory_id"],
                "reader": name,
                "export_directory": str(bundle),
            }
        )
    summary = (
        json.loads((root / "benchmark_summary.json").read_text())
        if (root / "benchmark_summary.json").exists()
        else {}
    )
    body = '<h1>成功任务完整轨迹</h1><p class="note">仅展示官方评测成功的任务。失败任务仍计入整体成功率；原始数据包保存完整日志、状态快照、模型请求与证据。没有保存模型私有思维过程。</p>'
    body += (
        f"<p>已完成 {summary.get('tasks_attempted', 0)} 题，"
        f"成功 {len(index)} 题，成功率 {summary.get('success_rate', 0):.2%}。</p>"
    )
    body += block("全量评测进度与统计", summary, True) + "<ul>" + "".join(links) + "</ul>"
    (destination / "index.html").write_text(page("成功任务完整轨迹", body))
    (destination / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Successful trajectories: {len(build_reader(args.directory))}")
