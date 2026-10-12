"""Disposable Mac measurements using production APIs; never selects P.

Run `prepare`, `smoke`, then `campaign` into a fresh ignored artifacts directory.
Only this harness instruments functions. Production modules are not edited.
"""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SEED = 20261011
WINDOW_DEADLINE = 2400
CHECKPOINT = Path("/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/gpu-readiness/artifacts/cpu-portable-smoke-001/step-1.pt")
EXECUTABLE = Path("/private/tmp/Stockfish-sf_17.1/src/stockfish")
BUNDLE = Path("/private/tmp/rl-chess-d08-production")


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    from chess_rl.annotation_benchmark import distribution
    return {**distribution(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "raw": values}


def paths():
    s = read(BUNDLE / "settings.json")
    return dict(bundle=BUNDLE, manifest_path=Path(s["manifest"]), games_path=Path(s["games_path"]),
                split_audit_path=Path(s["split_audit_path"]), prepared=Path(s["prepared_path"]),
                teacher_evidence=Path("/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/gpu-readiness/artifacts-historical-runs/d01/stockfish-provenance.json"))


def host():
    result = {}
    for name, command in {
        "cpu": ["sysctl", "hw.model", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize",
                "hw.perflevel0.name", "hw.perflevel0.physicalcpu", "hw.perflevel1.name", "hw.perflevel1.physicalcpu"],
        "memory_pressure": ["memory_pressure", "-Q"], "vm": ["vm_stat"],
        "swap": ["sysctl", "vm.swapusage"], "thermal": ["pmset", "-g", "therm"],
        "power": ["pmset", "-g", "batt"],
        "power_settings": ["pmset", "-g", "custom"],
    }.items():
        p = subprocess.run(command, capture_output=True, text=True)
        result[name] = dict(exit=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return result


def prepare(out):
    from chess_rl.data_manifest import DatasetManifest
    from chess_rl.stockfish_worker import replay_root, verify_teacher_bytes, verify_artifacts
    import torch
    import numpy as np
    import chess
    out.mkdir(parents=True, exist_ok=False)
    p = paths()
    manifest = DatasetManifest.load(p["manifest_path"])
    clock = time.perf_counter()
    verified = verify_artifacts(manifest, p["manifest_path"], p["games_path"], p["split_audit_path"], p["prepared"])
    teacher = verify_teacher_bytes(manifest, EXECUTABLE, p["teacher_evidence"])
    # Fixed training-only membership, hash-ranked within phase and colour.
    refs = [json.loads(line) for line in (BUNDLE / "selected.jsonl").read_text().splitlines()]
    selected = []
    for phase, quota in (("opening", 7), ("middlegame", 21), ("endgame", 4)):
        for colour in "wb":
            rows = [r for r in refs if r["split"] == "train" and r["phase"] == phase and r["rule_position"].split()[1] == colour]
            rows.sort(key=lambda r: hashlib.sha256((str(SEED) + json.dumps(r, sort_keys=True)).encode()).hexdigest())
            selected.extend(rows[:quota])
    assert len(selected) == 64
    wanted = {r["game_id"] for r in selected}
    games = {}
    with p["games_path"].open() as f:
        for line in f:
            g = json.loads(line)
            if g["game_id"] in wanted:
                games[g["game_id"]] = g
    for ref in selected:
        replay_root(ref, games[ref["game_id"]], manifest)
    save(out / "inputs.json", dict(references=selected, games=games))
    import shutil
    shutil.copyfile(CHECKPOINT, out / "checkpoint.pt")
    payload = torch.load(out / "checkpoint.pt", weights_only=False)
    save(out / "plan.json", dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        date=time.strftime("%Y-%m-%d",time.gmtime()), seed=SEED, hardware=host(), python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__, torch=str(torch.__version__), chess=chess.__version__,
        paths={k: str(v) for k, v in p.items()}, artifact_identities=verified, teacher=teacher,
        inputs_sha256=sha(out / "inputs.json"), source_labels_sha256=sha(BUNDLE / "labels.jsonl"),
        checkpoint=dict(original_path=str(CHECKPOINT), sha256=sha(out / "checkpoint.pt"),
                        checkpoint_id=payload["checkpoint_id"], model_config=payload["model_config"], provenance=payload["provenance"],
                        accepted_trained_P=False), preparation_seconds=time.perf_counter()-clock,
        storage="uncompressed JSONL canonical games/references/labels/attempts; immutable per-move JSON journals with dense masks/visits/policy; .pt checkpoint; no packed storage",
        settings=dict(label_workers=[1,2,4,8], stockfish_threads=1, timeout=30, max_attempts=2,
            label_positions=64, label_windows=3, persistent_worker_tested=False,
            training_batches=[1,4,16,32], torch_threads=[1,2,4,8], warmup_steps=3,
            training_windows=3, steps_per_window=6, decoded_cache_bytes=64*1024*1024,
            inference_batches=[1,4,16,32], inference_warmup=3, inference_windows=3, calls_per_window=10,
            selfplay_workers=[1,2,4,8], simulations=[16,32], ply_cap=512, torch_threads_per_worker=1,
            selfplay_windows=2, one_game_per_worker_per_window=True, journal=True,
            phases="production Settings defaults: temperature 1 before ply 30 then 0; noise .25; alpha .3",
            resource_policy=f"one intensive configuration at a time; RSS guard 5 GiB; ready barrier excludes process/model setup from timed windows; {WINDOW_DEADLINE}s timed window deadline",
            determinism="strict, float32, highest matmul precision, interop threads 1; standard Adam lr=1e-4 foreach=False fused=False")))
    print("Prepared fixed inputs and copied plumbing checkpoint", flush=True)


def ready(out, job):
    (out / f"ready-{job['id']}").touch()
    while not (out / job["barrier"]).exists():
        time.sleep(.05)


async def label_job(out, job):
    from chess_rl.data_manifest import DatasetManifest
    from chess_rl.stockfish_worker import AnnotationWorker, engine_attempt
    from chess_rl.pgn_replay import file_identity
    from chess_rl.annotation_benchmark import TransportClock
    import logging
    inputs, plan = read(out / "inputs.json"), read(out / "plan.json")
    manifest = DatasetManifest.load(Path(plan["paths"]["manifest_path"]))
    logs = []
    label_path = out / f"labels-{job['id']}.jsonl"
    attempt_path = out / f"attempts-{job['id']}.jsonl"
    provenance = dict(manifest_sha256=plan["artifact_identities"]["manifest"]["sha256"],
        games_sha256=plan["artifact_identities"]["games.jsonl"]["sha256"],
        samples_sha256=plan["artifact_identities"]["samples.jsonl"]["sha256"],
        overlap_audit_sha256=plan["artifact_identities"]["overlap-audit.json"]["sha256"],
        executable_sha256=plan["teacher"]["executable_identity"]["sha256"], embedded_nnue=plan["teacher"]["embedded_nnue"])
    with attempt_path.open("x") as af, label_path.open("x") as lf:
        def persist(f, value):
            f.write(json.dumps(value, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        def sink(record):
            persist(af, record)
            logs.append(record)
        async def attempt(state, trace):
            clock = TransportClock() if job.get("profile") else None
            logger = logging.getLogger("chess.engine")
            old_level = logger.level
            if clock:
                logger.setLevel(logging.DEBUG)
                logger.addHandler(clock)
            try:
                t = time.perf_counter()
                assert file_identity(EXECUTABLE) == plan["teacher"]["executable_identity"]
                trace["rehash_seconds"] = time.perf_counter()-t
                return await engine_attempt(state, manifest, EXECUTABLE, 30, trace)
            finally:
                if clock:
                    trace["timing"] = clock.metrics(time.perf_counter()-clock.started)
                    trace["timing"]["rehash_seconds"] = trace["rehash_seconds"]
                    logger.removeHandler(clock)
                    logger.setLevel(old_level)
        # A fresh throwaway worker/process warms the engine path, with no reused hash.
        warm = AnnotationWorker(manifest, provenance, attempt, lambda r: None)
        r = inputs["references"][job["indices"][0]]
        assert await warm.annotate(r, inputs["games"][r["game_id"]]) is not None
        worker = AnnotationWorker(manifest, provenance, attempt, sink)
        ready(out, job)
        start = time.perf_counter()
        latency, successes = [], 0
        for i in job["indices"]:
            r = inputs["references"][i]
            t = time.perf_counter()
            label = await worker.annotate(r, inputs["games"][r["game_id"]])
            if label is not None:
                persist(lf, label)
                successes += 1
            latency.append(time.perf_counter()-t)
        seconds = time.perf_counter()-start
    return dict(seconds=seconds, successful=successes, labels_per_second=successes/seconds,
                latency=stats(latency), attempts=len(logs), failures=sum(r["status"] != "success" for r in logs),
                timings=[r["returned"].get("timing") for r in logs] if job.get("profile") else None)


def train_job(out, job):
    import numpy as np
    import torch
    from chess_rl.runtime import configure
    from chess_rl.supervised import open_dataset
    from chess_rl.training.checkpoints import ModelConfig, DataOrder
    from chess_rl.training.trainer import supervised_step
    configure(threads=1)
    t = time.perf_counter()
    data = open_dataset(**paths(), split="train")
    open_seconds = time.perf_counter()-t
    t = time.perf_counter()
    data = data.freeze(cache_bytes=job.get("cache_bytes",64*1024*1024))
    freeze_seconds = time.perf_counter()-t
    save(out / f"setup-{job['id']}.json", dict(open_seconds=open_seconds, freeze_seconds=freeze_seconds,
        samples=len(data), source_identity=data.source_identity, cache_info=data.cache_info))
    ready(out, job)
    results = []
    configs = [(1,4)] if job.get("smoke") else [(th,b) for th in (1,2,4,8) for b in (1,4,16,32)]
    for th,b in configs:
        torch.set_num_threads(th)
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        model = ModelConfig().build()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        order = DataOrder("mac-benchmark", DataOrder.reference_hash(data), tuple(map(int,np.random.permutation(len(data)))),0,b,False,0)
        cursor=0
        def update(indices, current_order=None):
            a=time.perf_counter(); batch=data.batch(tuple(map(int,indices))); c=time.perf_counter()
            metric=supervised_step(model,opt,batch); trained=time.perf_counter()
            next_order=(replace(current_order,cursor=current_order.cursor+len(indices))
                        if current_order is not None else None)
            d=time.perf_counter()
            norm=float(torch.sqrt(sum(torch.sum(p.grad.detach().double()**2) for p in model.parameters())))
            return dict(seconds=d-a, loader_seconds=c-a, update_seconds=trained-c,
                        order_seconds=d-trained, gradient_norm=norm, metrics=asdict(metric)),next_order
        for _ in range(3):
            _,order=update(order.next_indices(),order)
        windows=[]; raw=[]
        for w in range(3):
            current=[]
            for _ in range(2 if job.get("smoke") else 6):
                measurement,order=update(order.next_indices(),order)
                current.append(measurement)
            raw.extend(current)
            seconds=sum(r["seconds"] for r in current)
            windows.append(dict(seconds=seconds,steps=len(current),steps_per_second=len(current)/seconds,samples_per_second=b*len(current)/seconds))
        # Exercise the actual production DataOrder wrap on all 9,000 references.
        # Start at its final full batch: a boundary probe, not a 9,000-sample epoch.
        boundary_order=replace(order,cursor=((len(data)-1)//b)*b)
        tail,_=update(boundary_order.next_indices(),boundary_order)
        boundary_order=replace(boundary_order,cursor=len(data))
        assert not boundary_order.next_indices()
        a=time.perf_counter()
        boundary_order=replace(boundary_order,order=tuple(map(int,np.random.permutation(len(data)))),cursor=0,epoch=1)
        reshuffle_seconds=time.perf_counter()-a
        head,_=update(boundary_order.next_indices(),boundary_order)
        result=dict(threads=th,batch=b,raw=raw,windows=windows,non_finite_updates=0,
            gradient_norms=stats([r["gradient_norm"] for r in raw]),
            step_seconds=stats([r["seconds"] for r in raw]), loader_seconds=stats([r["loader_seconds"] for r in raw]),
            boundary=dict(tail=tail,reshuffle_seconds=reshuffle_seconds,head=head,full_epoch_executed=False), cache_info=data.cache_info)
        results.append(result)
        save(out / f"train-t{th}-b{b}-{job['id']}.json",result)
        print(json.dumps(dict(workload="training",threads=th,batch=b,samples_per_second=statistics.mean(w["samples_per_second"] for w in windows))),flush=True)
        del model,opt
    if job.get("training_profile"):
        import cProfile
        torch.set_num_threads(4);torch.manual_seed(SEED)
        model=ModelConfig().build()
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False,fused=False)
        indices=order.order[:32]
        for _ in range(3):update(indices)
        profiler=cProfile.Profile()
        profiler.enable()
        for _ in range(3):update(indices)
        profiler.disable()
        profiler.dump_stats(str(out / f"profile-{job['id']}.pstats"))
    return dict(configurations=results,cache_bytes=job.get("cache_bytes",64*1024*1024))


def inference_job(out,job):
    import torch
    from chess_rl.runtime import configure
    from chess_rl.stockfish_worker import replay_root
    from chess_rl.data_manifest import DatasetManifest
    from chess_rl.search.real_evaluator import RealEvaluator
    configure(threads=1)
    inputs=read(out / "inputs.json")
    manifest=DatasetManifest.load(paths()["manifest_path"])
    states=[replay_root(r,inputs["games"][r["game_id"]],manifest) for r in inputs["references"]]
    evaluator=RealEvaluator(out / "checkpoint.pt")
    ready(out,job)
    results=[]
    for th in ((1,) if job.get("smoke") else (1,2,4,8)):
        torch.set_num_threads(th)
        for batch in ((1,4) if job.get("smoke") else (1,4,16,32)):
            selected=[states[(SEED + i*len(states)//batch) % len(states)] for i in range(batch)]
            for _ in range(3): evaluator(selected)
            windows=[];raw=[]
            for window in range(3):
                values=[]
                for call_index in range(3 if job.get("smoke") else 10):
                    if job.get("rotate_inputs"):
                        offset=SEED+(window*10+call_index)*len(states)//30
                        selected=[states[(offset+i*len(states)//batch)%len(states)] for i in range(batch)]
                    t=time.perf_counter(); evaluator(selected); values.append(time.perf_counter()-t)
                raw.extend(values);seconds=sum(values)
                windows.append(dict(seconds=seconds,calls_per_second=len(values)/seconds,samples_per_second=batch*len(values)/seconds))
            results.append(dict(threads=th,batch=batch,latency=stats(raw),windows=windows))
            print(json.dumps(dict(workload="inference",threads=th,batch=batch,latency_median=statistics.median(raw))),flush=True)
    return dict(configurations=results,rotate_inputs=job.get("rotate_inputs",False),
                scope="production RealEvaluator batch microbenchmark; no end-to-end batched self-play")


def selfplay_job(out,job):
    import torch
    import chess
    from chess_rl.runtime import configure
    from chess_rl.selfplay import Session, Settings
    from chess_rl.evaluation.matches import Opening, digest
    from chess_rl.adapter import make_state
    configure(threads=1)
    opening=Opening.from_state("mac-standard-start",make_state(),source="Mac profiling fixed standard initial position")
    a=time.perf_counter()
    session=Session.create(out / f"game-{job['id']}",out / "checkpoint.pt",opening,
        settings=Settings(job["simulations"],512),seed=job["seed"],opening_selection_seed=SEED,journal=True)
    setup=time.perf_counter()-a
    session.evaluator([make_state()])  # warm up model; no training/RNG update
    samples=[];totals=Counter(); stack=[]
    # Exclusive category timers are installed only in profiler runs. Normal
    # throughput jobs leave the evaluator and production functions unchanged.
    def timed(category, fn):
        def wrapped(*args,**kwargs):
            frame=[time.perf_counter(),0.0];stack.append(frame)
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-frame[0];stack.pop()
                totals[category]+=elapsed-frame[1]
                if stack:stack[-1][1]+=elapsed
        return wrapped
    profile=job.get("profile",False)
    if profile:
        import chess_rl.search.real_evaluator as re
        original=re.encode_position
        re.encode_position=timed("feature_encoding",original)
        model=session.evaluator.checkpoint.model
        model.forward=timed("neural_forward",model.forward)
        session.evaluator=timed("evaluator_other",session.evaluator)
    original_publish=session._publish
    if profile: original_publish=timed("record_publication",original_publish)
    last=[0.0]; previous=Counter()
    def publish(record,**kwargs):
        start=time.perf_counter()
        original_publish(record,**kwargs)
        if kwargs.get("position") is not None:
            now=time.perf_counter()
            row=dict(ply=len(record["moves_uci"]),move_seconds=now-last[0],publication_seconds=now-start)
            if profile:
                delta=totals-previous
                row["exclusive_seconds"]=dict(delta)
                row["tree_board_history_and_record_assembly_seconds"]=max(0,row["move_seconds"]-sum(delta.values()))
                previous.clear();previous.update(totals)
            samples.append(row);last[0]=time.perf_counter()
    session._publish=publish
    ready(out,job)
    started=time.perf_counter();last[0]=started
    if profile:
        import cProfile
        profiler=cProfile.Profile();profiler.enable()
    snapshot=session.run(stop_after_moves=job.get("stop_after_moves"))
    if profile:
        profiler.disable();profiler.dump_stats(str(out / f"profile-{job['id']}.pstats"))
    seconds=time.perf_counter()-started
    game=snapshot["game"]
    n=len(game["positions"]); sims=sum(p["completed_simulations"] for p in game["positions"])
    buckets={}
    for name,lower,upper in (("early",1,32),("middle",65,128),("late",257,512)):
        rows=[r for r in samples if lower<=r["ply"]<=upper]
        if rows:
            buckets[name]=dict(moves=len(rows),move_seconds=stats([r["move_seconds"] for r in rows]),
                publication_seconds=stats([r["publication_seconds"] for r in rows]),
                exclusive_seconds={k:sum(r.get("exclusive_seconds",{}).get(k,0) for r in rows) for k in totals},
                residual_seconds=sum(r.get("tree_board_history_and_record_assembly_seconds",0) for r in rows))
    return dict(seconds=seconds,setup_seconds=setup,status=game["status"],outcome=game.get("outcome"),
        plies=n,simulations=sims,simulations_per_second=sims/seconds,move_samples=samples,buckets=buckets,
        exclusive_seconds=dict(totals),final_copy_and_run_overhead_seconds=seconds-sum(r["move_seconds"] for r in samples),
        trajectory_sha256=digest(dict(moves=game["moves_uci"],status=game["status"],outcome=game.get("outcome"))),
        checkpoint_sha256=session.configuration["checkpoint"]["sha256"],seed=job["seed"],profile=profile)


def process_sample(pids):
    p=subprocess.run(["ps","-axo","pid=,ppid=,rss=,%cpu="],capture_output=True,text=True)
    if p.returncode:return dict(error=p.stderr)
    rows=[]
    for line in p.stdout.splitlines():
        a=line.split()
        if len(a)==4:rows.append((int(a[0]),int(a[1]),int(a[2])*1024,float(a[3])))
    tree=set(pids)
    while True:
        children={pid for pid,parent,_,_ in rows if parent in tree}
        if children<=tree:break
        tree.update(children)
    selected=[r for r in rows if r[0] in tree]
    return dict(rss_bytes=sum(r[2] for r in selected),cpu_percent=sum(r[3] for r in selected),processes=len(selected))


def history_job(out,job):
    """Legal long-history search probe; explicitly not completed-game throughput."""
    import chess
    import cProfile
    import numpy as np
    from chess_rl.runtime import configure
    from chess_rl.adapter import make_state
    from chess_rl.search.real_evaluator import RealEvaluator
    from chess_rl.search.puct import Node, _fresh_root
    from chess_rl.search.runner import search
    configure(threads=1)
    # The Stage 1 legal quiet trajectory algorithm; only fixture construction.
    rng=random.Random(761);state=make_state();moves=[]
    for ply in range(512):
        choices=list(state.board.legal_moves);rng.shuffle(choices)
        prefer_pawn=state.board.halfmove_clock>=70
        choices.sort(key=lambda m:(state.board.is_capture(m),
            (state.board.piece_type_at(m.from_square)==chess.PAWN)!=prefer_pawn))
        for move in choices:
            board=state.board.copy(stack=True);board.push(move)
            if board.outcome(claim_draw=False) is None and not board.is_repetition(3) and not board.is_fifty_moves():
                state.board.push(move);moves.append(move.uci());break
        else:raise AssertionError(f"long fixture terminated at {ply}")
    save(out / "history-probe-moves.json",moves)
    evaluator=RealEvaluator(out / "checkpoint.pt")
    ready(out,job)
    results=[]
    for sims in (16,32):
        for ply in (0,96,256,480):
            state=make_state(moves_uci=moves[:ply])
            a=time.perf_counter();owner=Node(state)
            admission_seconds=time.perf_counter()-a
            def call():
                # Match Session.run after its first move: trusted search-owned
                # root copying, without repeating public full-history admission.
                return search(_fresh_root(owner),sims,evaluator=evaluator)
            call()  # fresh root on every warm-up and measured search
            times=[]
            for _ in range(3):
                a=time.perf_counter();result=call();times.append(time.perf_counter()-a)
                assert result.completed_simulations==sims
            profiler=cProfile.Profile()
            profiler.runcall(call)
            profiler.dump_stats(str(out / f"profile-history-s{sims}-p{ply}.pstats"))
            forwards=sum(e.callcount for e in profiler.getstats()
                         if hasattr(e.code,"co_filename") and e.code.co_filename.endswith("model/network.py")
                         and e.code.co_name=="forward")
            results.append(dict(simulations=sims,history_plies=ply,admission_seconds=admission_seconds,
                                search_seconds=stats(times),neural_forwards_per_search=forwards))
    return dict(configurations=results,scope="fixed legal long-trajectory MCTS probe; board/material vary along trajectory; no publication or completed-game throughput")


def group(out,name,jobs):
    completed=out / f"group-{name}.json"
    if completed.exists():
        previous=read(completed)
        assert previous["jobs"]==jobs,"completed benchmark settings differ"
        return previous
    assert not (out / f"start-{name}").exists()
    procs=[];handles=[]
    before=host()
    for i,job in enumerate(jobs):
        job={**job,"id":f"{name}-{i}","barrier":f"start-{name}"}
        save(out / f"job-{job['id']}.json",job)
        f=(out / f"stdout-{job['id']}.log").open("x");handles.append(f)
        procs.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--output",str(out),"--job",job["id"]],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT))
    t=time.perf_counter();samples=[];setup_samples=[]
    def cleanup():
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:p.wait()
        for f in handles:f.close()
    try:
        while not all((out / f"ready-{name}-{i}").exists() for i in range(len(jobs))):
            if any(p.poll() is not None for p in procs):
                raise RuntimeError(f"worker failed before ready: {name}; inspect stdout logs")
            s=process_sample([p.pid for p in procs]);s["elapsed"]=time.perf_counter()-t
            setup_samples.append(s)
            if s.get("rss_bytes",0)>5*1024**3:raise MemoryError("setup process-tree RSS guard 5 GiB")
            if time.perf_counter()-t>600:raise TimeoutError(f"setup deadline: {name}")
            time.sleep(.5)
    except BaseException:
        save(out / f"failed-setup-{name}.json",dict(resources=setup_samples))
        cleanup()
        raise
    setup_seconds=time.perf_counter()-t
    started=time.perf_counter();(out / f"start-{name}").touch()
    try:
        while any(p.poll() is None for p in procs):
            s=process_sample([p.pid for p in procs]);s["elapsed"]=time.perf_counter()-started;samples.append(s)
            if s.get("rss_bytes",0)>5*1024**3:raise MemoryError("process-tree RSS guard 5 GiB")
            if time.perf_counter()-started>WINDOW_DEADLINE:raise TimeoutError(f"timed window deadline {WINDOW_DEADLINE}s")
            time.sleep(.2)
        exits=[p.wait() for p in procs]
        assert not any(exits),f"worker failure {name}; see stdout logs"
        results=[read(out / f"result-{name}-{i}.json") for i in range(len(jobs))]
        end_to_end=time.perf_counter()-started
        # Workers signal completion with measured body duration before exiting;
        # barrier-release scheduling is <=50ms; process teardown is separate.
        body=max(r["seconds"] for r in results) if all("seconds" in r for r in results) else end_to_end
        report=dict(name=name,jobs=jobs,results=results,body_seconds=body,group_wall_seconds=end_to_end,
            timed_window_deadline_seconds=WINDOW_DEADLINE,
            harness_sha256=sha(Path(__file__).resolve()),
            launch_and_setup_seconds=setup_seconds,resources=samples,
            setup_resources=setup_samples,
            peak_setup_tree_rss_bytes=max((r.get("rss_bytes",0) for r in setup_samples),default=0),
            peak_entire_tree_rss_bytes=max((r.get("rss_bytes",0) for r in setup_samples+samples),default=0),
            peak_tree_rss_bytes=max((r.get("rss_bytes",0) for r in samples),default=0),
            mean_sampled_cpu_percent=statistics.mean(r.get("cpu_percent",0) for r in samples) if samples else None,
            before=before,after=host())
        save(out / f"group-{name}.json",report)
        print(json.dumps(dict(group=name,seconds=body,peak_rss_bytes=report["peak_tree_rss_bytes"],
            statuses=[r.get("status") for r in results],plies=[r.get("plies") for r in results])),flush=True)
        return report
    finally:
        cleanup()


def label_equivalence(out,groups):
    canonical={};compared=0; differences=[]
    for g in groups:
        for i in range(len(g["jobs"])):
            for line in (out / f"labels-{g['name']}-{i}.jsonl").read_text().splitlines():
                label=json.loads(line)
                # Entire persisted label except explicitly timing-dependent telemetry.
                label["returned_search"].pop("time",None);label["returned_search"].pop("nps",None)
                key=label["game_id"]+":"+str(label["sample_ply"])
                if key in canonical:
                    compared+=1
                    if canonical[key]!=label:differences.append(key)
                else:canonical[key]=label
    result=dict(unique_positions=len(canonical),comparisons=compared,differences=differences,
        excluded_fields=["returned_search.time","returned_search.nps"],all_other_label_fields_exact=not differences,
        persistent_worker_tested=False)
    save(out / "label-equivalence.json",result)
    assert not differences,"Scheduling changed non-timing label fields"


def run(out,smoke=False):
    prefix="smoke" if smoke else "campaign"
    label_groups=[]
    for workers in ((1,) if smoke else (1,2,4,8)):
        for window in range(1 if smoke else 3):
            order=list(range(4 if smoke else 64));random.Random(SEED+window).shuffle(order)
            label_groups.append(group(out,f"{prefix}-label-w{workers}-r{window}",
                [dict(kind="label",indices=order[i::workers]) for i in range(workers)]))
    if not smoke:label_equivalence(out,label_groups)
    group(out,f"{prefix}-train",[dict(kind="train",smoke=smoke)])
    group(out,f"{prefix}-inference",[dict(kind="inference",smoke=smoke)])
    for sims in ((16,) if smoke else (16,32)):
        for workers in ((1,) if smoke else (1,2,4,8)):
            for window in range(1 if smoke else 2):
                group(out,f"{prefix}-selfplay-s{sims}-w{workers}-r{window}",
                    [dict(kind="selfplay",simulations=sims,seed=SEED+100*window+i,
                          **({"stop_after_moves":8} if smoke else {})) for i in range(workers)])
    if not smoke:
        group(out,"profile-label",[dict(kind="label",indices=[4*i for i in range(16)],profile=True)])
        for sims in (16,32):
            group(out,f"profile-selfplay-s{sims}",[dict(kind="selfplay",simulations=sims,seed=SEED,profile=True)])
    save(out / f"{prefix}-complete.json",dict(completed=True,hardware=host(),harness_sha256=sha(Path(__file__).resolve())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","smoke","campaign","selfplay32","stop","stop-setup","supplement","worker"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--job")
    args=p.parse_args();out=args.output.resolve()
    if args.mode=="stop-setup":
        import shlex
        import signal
        rows=subprocess.check_output(["ps","-axo","pid=,args="],text=True)
        matches=[]
        for line in rows.splitlines():
            pid,_,command=line.strip().partition(" ")
            try:parts=shlex.split(command)
            except ValueError:continue
            if (len(parts)==7 and Path(parts[0]).name.startswith("python")
                and parts[1]==str(Path(__file__).resolve()) and parts[2]=="worker"
                and parts[3]=="--output" and Path(parts[4]).resolve()==out
                and parts[5]=="--job" and parts[6].startswith("campaign-selfplay-s32-")):
                job=read(out / f"job-{parts[6]}.json")
                assert not (out/job['barrier']).exists(),"worker is already in a timing window"
                matches.append(int(pid))
        save(out / "deadline-extension-setup-cleanup.json",dict(pids=matches))
        for pid in matches:os.kill(pid,signal.SIGTERM)
        print(f"Stopped {len(matches)} disposable setup worker(s)",flush=True)
    elif args.mode=="stop":
        # Stop only this output directory's campaign orchestrator. Its timed
        # group finally block terminates its disposable workers and keeps logs.
        import shlex
        import signal
        rows=subprocess.check_output(["ps","-axo","pid=,args="],text=True)
        matches=[]
        for line in rows.splitlines():
            pid,_,command=line.strip().partition(" ")
            try:parts=shlex.split(command)
            except ValueError:continue
            if (len(parts)==5 and Path(parts[0]).name.startswith("python")
                and parts[1] in (str(Path(__file__).resolve()),"scripts/profile_mac.py")
                and parts[2]=="campaign" and parts[3]=="--output"
                and (ROOT/parts[4]).resolve()==out):matches.append(int(pid))
        assert len(matches)==1,f"expected exactly one matching campaign, found {len(matches)}"
        save(out / "deadline-extension-stop.json",dict(pid=matches[0],reason="32-simulation waves need a 2400s bound; 16-simulation sweep retained",utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
        os.kill(matches[0],signal.SIGINT)
        print("Stopped the matching disposable campaign; existing outputs retained",flush=True)
    elif args.mode=="selfplay32":
        for workers in (1,2,4,8):
            for window in range(2):
                group(out,f"campaign-selfplay-s32-w{workers}-r{window}",
                    [dict(kind="selfplay",simulations=32,seed=SEED+100*window+i) for i in range(workers)])
        group(out,"profile-label",[dict(kind="label",indices=[4*i for i in range(16)],profile=True)])
        for sims in (16,32):
            group(out,f"profile-selfplay-s{sims}",[dict(kind="selfplay",simulations=sims,seed=SEED,profile=True)])
        save(out / "campaign-complete.json",dict(completed=True,hardware=host(),harness_sha256=sha(Path(__file__).resolve()),timed_window_deadline_seconds=WINDOW_DEADLINE))
    elif args.mode=="prepare":prepare(out)
    elif args.mode=="worker":
        job=read(out / f"job-{args.job}.json")
        if job["kind"]=="label":result=asyncio.run(label_job(out,job))
        elif job["kind"]=="train":result=train_job(out,job)
        elif job["kind"]=="inference":result=inference_job(out,job)
        elif job["kind"]=="history":result=history_job(out,job)
        else:result=selfplay_job(out,job)
        save(out / f"result-{args.job}.json",result)
    elif args.mode=="supplement":
        group(out,"supplement-train-cache0",[dict(kind="train",cache_bytes=0,training_profile=True)])
        group(out,"supplement-inference",[dict(kind="inference",rotate_inputs=True)])
        group(out,"supplement-history",[dict(kind="history")])
        group(out,"profile-selfplay-long-s32",[dict(kind="selfplay",simulations=32,seed=SEED+100,profile=True)])
        order=list(range(64));random.Random(SEED).shuffle(order)
        group(out,"supplement-label-anchor",[dict(kind="label",indices=order)])
        save(out / "supplement-complete.json",dict(completed=True,harness_sha256=sha(Path(__file__).resolve())))
    else:run(out,args.mode=="smoke")


if __name__=="__main__":main()
