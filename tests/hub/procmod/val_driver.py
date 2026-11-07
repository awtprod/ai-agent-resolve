import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import val_recipes

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.recipes.engine import advance
from aar.recipes.worker import run_worker

action, db, artifacts, actor_s, recipe_node = sys.argv[1:6]
kind, _, ident = actor_s.partition(":")
actor = Actor(kind, ident)
go = os.environ.get("VAL_GO")
if go:
    while not os.path.exists(go):
        time.sleep(0.001)
disp_root = os.environ.get("VAL_DISPATCH")
dispatcher = val_recipes.FileDispatcher(disp_root) if disp_root else None
log = EventLog(db)
try:
    if action == "advance":
        for _ in range(int(os.environ.get("VAL_PASSES", "1"))):
            advance(log, recipe_node, actor=actor, dispatcher=dispatcher)
    elif action == "worker":
        store = ArtifactStore(artifacts, log)
        run_worker(
            log,
            store,
            actor,
            dispatcher=dispatcher,
            once=os.environ.get("VAL_ONCE") == "1",
            lease_s=int(os.environ.get("VAL_LEASE", "60")),
            heartbeat_s=int(os.environ.get("VAL_HB", "15")),
            poll_s=float(os.environ.get("VAL_POLL", "0.1")),
            install_signals=True,
        )
finally:
    log.close()
