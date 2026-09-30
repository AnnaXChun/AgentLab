from sqlalchemy import select
from sqlalchemy.orm import Session

from experiment_infra.core.database import ObjectIndex, ProvenanceEdge
from experiment_infra.core.schema import OBJECT_IDS, OBJECT_TYPES
from packages.core.events.recorder import PAYLOAD_LIMIT, Recorder
from packages.core.events.store import GENESIS, canonical, event_hash
from packages.core.models.database import EpisodeRow
from packages.core.models.schemas import Episode, Event
from packages.core.models.schemas import EventType as T


class Ledger:
    """Reuse the canonical event store. Indexes and entity events commit atomically."""

    def __init__(self, store, artifacts):
        self.store, self.artifacts = store, artifacts

    def recorder(self, experiment_id):
        return Recorder(
            self.store,
            self.artifacts,
            experiment_id,
            self.store.episode(experiment_id).root_branch_id,
        )

    def create_partition(self, experiment):
        ep = Episode(
            episode_id=experiment.experiment_id,
            goal=experiment.goal,
            domain_pack="experiment-infrastructure",
            agent_id=experiment.agent_backend,
            metadata={"public_schema": "2.0", "kind": "Experiment"},
        )
        self.store.create_episode(
            ep,
            Event(
                episode_id=ep.episode_id,
                branch_id=ep.root_branch_id,
                event_type=T.EPISODE_STARTED,
                output={"episode": ep.model_dump(mode="json")},
            ),
        )

    async def record(
        self,
        experiment_id,
        event_type,
        objects=(),
        links=(),
        *,
        input=None,
        output=None,
        status="SUCCESS",
    ):
        encoded = [
            {
                "id": getattr(obj, OBJECT_IDS[type(obj).__name__]),
                "kind": type(obj).__name__,
                "data": obj.model_dump(mode="json"),
            }
            for obj in objects
        ]
        body = {
            **(output or {}),
            "objects": encoded,
            "provenance_links": [list(link) for link in links],
        }
        rec = self.recorder(experiment_id)
        refs = []
        if len(canonical(body)) > PAYLOAD_LIMIT:
            artifact = await rec.artifact(
                canonical(body), "application/json", {"purpose": "entity-event-payload"}
            )
            refs.append(artifact.artifact_id)
            body = {"$artifact": artifact.artifact_id}
        request = input or {}
        if len(canonical(request)) > PAYLOAD_LIMIT:
            artifact = await rec.artifact(
                canonical(request), "application/json", {"purpose": "entity-event-input"}
            )
            refs.append(artifact.artifact_id)
            request = {"$artifact": artifact.artifact_id}
        with Session(self.store.engine) as session, session.begin():
            session.scalar(
                select(EpisodeRow).where(EpisodeRow.episode_id == experiment_id).with_for_update()
            )
            event = self.store._append(
                session,
                Event(
                    episode_id=experiment_id,
                    branch_id=rec.branch_id,
                    event_type=event_type,
                    input=request,
                    output=body,
                    artifact_refs=refs,
                    status=status,
                ),
            )
            for obj in encoded:
                previous = session.get(ObjectIndex, obj["id"])
                if previous:
                    if obj["kind"] in {
                        "Artifact",
                        "Evidence",
                        "VerificationResult",
                        "Claim",
                        "TrajectoryStep",
                        "Snapshot",
                        "ActionEvent",
                        "CounterfactualLink",
                    }:
                        raise ValueError(
                            "Immutable object identity already registered; create a new version"
                        )
                    if previous.experiment_id != experiment_id or previous.kind != obj["kind"]:
                        raise ValueError("Object identity cannot change scope or type")
                else:
                    session.add(
                        ObjectIndex(
                            object_id=obj["id"],
                            experiment_id=experiment_id,
                            kind=obj["kind"],
                            created_event_id=event.event_id,
                        )
                    )
            session.flush()
            edges = {
                (e.source_id, e.target_id, e.relation)
                for e in session.scalars(select(ProvenanceEdge))
            }
            for source, target, relation in links:
                src, dst = session.get(ObjectIndex, source), session.get(ObjectIndex, target)
                if src is None or dst is None or dst.experiment_id != experiment_id:
                    raise ValueError("Unresolved provenance endpoint")
                # Reject self-links and cycles, including links in this transaction.
                frontier, reached = [target], set()
                while frontier:
                    current = frontier.pop()
                    if current in reached:
                        continue
                    reached.add(current)
                    frontier.extend(t for s, t, _ in edges if s == current)
                if source in reached:
                    raise ValueError("Provenance must be acyclic")
                key = (source, target, relation)
                if key not in edges:
                    session.add(
                        ProvenanceEdge(
                            source_id=source,
                            target_id=target,
                            relation=relation,
                            created_event_id=event.event_id,
                        )
                    )
                    edges.add(key)
            return event

    def locate(self, object_id, kind=None):
        with Session(self.store.engine) as session:
            row = session.get(ObjectIndex, object_id)
            if row is None or (kind and row.kind != kind):
                raise KeyError(object_id)
            return row.experiment_id

    async def objects(self, experiment_id):
        rec = self.recorder(experiment_id)
        values, before = {}, GENESIS
        for event in self.store.events(experiment_id):
            if event.state_before_hash != before or event_hash(event) != event.state_after_hash:
                raise ValueError("Canonical experiment history integrity failed")
            before = event.state_after_hash
            output = await rec.resolve(event.output)
            for obj in output.get("objects", []):
                values[obj["id"]] = OBJECT_TYPES[obj["kind"]].model_validate(obj["data"])
        return values

    async def get(self, object_id, kind=None):
        experiment_id = self.locate(object_id, kind)
        return (await self.objects(experiment_id))[object_id]

    async def get_provenance(self, object_id):
        self.locate(object_id)
        with Session(self.store.engine) as session:
            edges = [
                (e.source_id, e.target_id, e.relation)
                for e in session.scalars(select(ProvenanceEdge))
            ]

        def traverse(reverse):
            found, frontier = set(), [object_id]
            while frontier:
                current = frontier.pop()
                for source, target, _ in edges:
                    start, end = (target, source) if reverse else (source, target)
                    if start == current and end not in found:
                        found.add(end)
                        frontier.append(end)
            return found

        ancestors, descendants = traverse(True), traverse(False)
        ids = ancestors | descendants | {object_id}
        cache = {}
        nodes = {}
        for oid in sorted(ids):
            eid = self.locate(oid)
            if eid not in cache:
                cache[eid] = await self.objects(eid)
            obj = cache[eid][oid]
            nodes[oid] = {"kind": type(obj).__name__, "object": obj.model_dump(mode="json")}
        return {
            "schema_version": "2.0",
            "object_id": object_id,
            "ancestors": sorted(ancestors),
            "descendants": sorted(descendants),
            "nodes": nodes,
            "edges": [
                {"source": s, "target": t, "relation": r}
                for s, t, r in edges
                if s in ids and t in ids
            ],
        }

    async def get_ancestors(self, object_id):
        return (await self.get_provenance(object_id))["ancestors"]

    async def get_descendants(self, object_id):
        return (await self.get_provenance(object_id))["descendants"]
