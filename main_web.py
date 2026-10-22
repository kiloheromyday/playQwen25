import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import uvicorn

from qwen25.config import QwenConfig
from qwen25.data import load_text_documents, split_documents
from qwen25.evaluation import score_documents
from qwen25.inference import generate_text
from qwen25.model import QwenForCausalLM
from qwen25.tokenizer import SimpleCharTokenizer


ROOT = Path(__file__).resolve().parent


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=10000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)
    history: bool = True
    max_new_tokens: int = Field(default=48, ge=1, le=96)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=2147483647)


class EvalRequest(BaseModel):
    custom_text: str = Field(default="", max_length=20000)


class LocalLab:
    def __init__(self, checkpoint, device="cpu"):
        self.path = Path(checkpoint).resolve()
        self.device = torch.device(device)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        self.metadata = payload["metadata"]
        self.config = QwenConfig(**self.metadata["config"])
        self.tokenizer = SimpleCharTokenizer.from_dict(self.metadata["tokenizer"])
        self.model = QwenForCausalLM(self.config).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        self.step = payload["step"]
        self.lock = threading.Lock()
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.eval_path = ROOT / "artifacts/web/latest_evaluation.json"
        self.latest_report = None
        if self.eval_path.exists():
            saved = json.loads(self.eval_path.read_text(encoding="utf-8"))
            if saved.get("checkpoint_sha256") == self.fingerprint:
                self.latest_report = saved
        # Warm up CUDA outside user-visible latency measurements.
        generate_text(self.model, self.tokenizer, "训练", self.device, max_new_tokens=2, temperature=0)

    def info(self):
        return {"name": "Qwen2.5-style · Tiny", "checkpoint": self.path.name,
                "checkpoint_directory": self.path.parent.name, "sha256": self.fingerprint,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "step": self.step, "device": str(self.device), "precision": "FP32",
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "config": asdict(self.config), "tokenizer": "字符级教学词表",
                "instruction_tuned": False}

    def chat(self, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "模型正在执行其他任务，请稍后重试")
        try:
            if request.messages[-1].role != "user":
                raise ValueError("最后一条消息必须是用户输入")
            # This base LM has no chat template: history is plain text continuation.
            messages = request.messages if request.history else request.messages[-1:]
            prompt = "\n".join(m.content for m in messages)
            result = generate_text(self.model, self.tokenizer, prompt, self.device,
                                   request.max_new_tokens, request.temperature, request.top_p,
                                   request.repetition_penalty, request.seed)
            result["context_mode"] = "plain_text_history" if request.history else "last_input"
            return result
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        finally:
            self.lock.release()

    def evaluate(self, request):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "模型正在执行其他任务，请稍后重试")
        try:
            started = time.perf_counter()
            documents = load_text_documents(ROOT / "data/tiny_corpus")
            train, valid = split_documents(documents, self.metadata.get("valid_ratio", 0.25),
                                           self.metadata.get("seed", 0))
            sets = {"train": train, "validation": valid}
            if request.custom_text.strip():
                sets["custom"] = [request.custom_text.strip()]
            trained = {name: score_documents(self.model, self.tokenizer, docs, self.device)
                       for name, docs in sets.items() if docs}
            cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(42)
                baseline = QwenForCausalLM(self.config).to(self.device).eval()
            random_scores = {name: score_documents(baseline, self.tokenizer, docs, self.device)
                             for name, docs in sets.items() if docs}
            del baseline
            probes = []
            for name, docs in (("训练文本续写", train), ("验证文本续写", valid)):
                if not docs:
                    continue
                sentence = docs[0].splitlines()[0]
                prefix, reference = sentence[:8], sentence[8:24]
                generated = generate_text(self.model, self.tokenizer, prefix, self.device,
                                          max_new_tokens=16, temperature=0, repetition_penalty=1, seed=0)
                probes.append({"source": name, "prompt": prefix, "reference": reference,
                               "exact_match": generated["text"] == reference, **generated})
            report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "checkpoint_sha256": self.fingerprint, "model": self.info(),
                      "trained": trained, "random_baseline": random_scores, "probes": probes,
                      "corpus_sha256": hashlib.sha256(json.dumps(documents, ensure_ascii=False).encode()).hexdigest(),
                      "data_source": "当前项目教学语料；按文档切分，默认 seed=0 / valid_ratio=0.25",
                      "elapsed_seconds": time.perf_counter() - started,
                      "protocol": {"context": self.config.max_position_embeddings,
                                   "overlap": 1, "include_eos": True, "random_seed": 42,
                                   "generation": "greedy, repetition_penalty=1, 16 tokens",
                                   "timing": "CUDA synchronized, warm model, batch=1; excludes HTTP/browser",
                                   "scope": "教学诊断；基础模型续写，不代表通用问答或官方 Qwen 能力"}}
            self.eval_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.latest_report = report
            return report
        finally:
            self.lock.release()


def create_app(lab):
    app = FastAPI(title="Qwen Local Lab")

    @app.get("/api/info")
    def info():
        return lab.info()

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        return lab.chat(request)

    @app.post("/api/evaluate")
    def evaluate(request: EvalRequest):
        return lab.evaluate(request)

    @app.get("/api/evaluation")
    def evaluation():
        return lab.latest_report

    @app.get("/api/evaluation/download")
    def download():
        if lab.latest_report is None:
            raise HTTPException(404, "请先运行评测")
        return FileResponse(lab.eval_path, filename="qwen-local-evaluation.json", media_type="application/json")

    @app.get("/api/training")
    def training():
        path = lab.path.parent / "results.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    icons = ROOT / "node_modules/lucide/dist/umd/lucide.min.js"

    @app.get("/icons.js")
    def icon_script():
        if not icons.exists():
            raise HTTPException(404, "请先运行 npm install")
        return FileResponse(icons, media_type="application/javascript")

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


def main():
    parser = argparse.ArgumentParser(description="Local chat and evaluation for the teaching checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/local_smoke_20261021/checkpoint.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    checkpoint = ROOT / args.checkpoint
    lab = LocalLab(checkpoint, args.device)
    port = args.port
    while port <= args.port + 20:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                break
            except OSError:
                port += 1
    else:
        raise RuntimeError("没有可用端口")
    print(f"Local Lab: http://127.0.0.1:{port}", flush=True)
    uvicorn.run(create_app(lab), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
