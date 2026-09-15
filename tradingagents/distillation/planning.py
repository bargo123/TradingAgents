"""Deterministic, structure-aware source packet planning."""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from .errors import SourcePacketTooLargeError
from .models import SourcePacket, SourcePlan, canonical_hash

@dataclass(frozen=True, slots=True)
class PlannerConfig:
    max_blocks: int = 8
    max_chars: int = 12000
    max_estimated_tokens: int = 3000
    min_score: int = 1
    def __post_init__(self):
        if min(self.max_blocks,self.max_chars,self.max_estimated_tokens)<1: raise ValueError("planner bounds must be positive")

class SourcePacketPlanner:
    @staticmethod
    def plan(source, topics, config: PlannerConfig | None = None) -> SourcePlan:
        config=config or PlannerConfig(); terms=tuple(sorted(set(re.findall(r"[a-z0-9]+", " ".join(topics).lower()))))
        blocks=sorted(tuple(source.blocks()), key=lambda b:(b.ref.document_id,b.reading_order,b.ref.chunk_id))
        selected=[b for b in blocks if sum(b.text.lower().count(t) for t in terms)>=config.min_score] if terms else blocks
        selected_ids = {id(b) for b in selected}
        # Retain immediately adjacent table/caption and equation/definition companions.
        selected = [b for i, b in enumerate(blocks) if id(b) in selected_ids or (i and id(blocks[i-1]) in selected_ids and str(b.content_type).upper() in {"TABLE", "FIGURE_CAPTION", "EQUATION", "DEFINITION"})]
        packets=[]; diagnostics=[]; i=0
        while i<len(selected):
            first=selected[i]; group=[first]; chars=len(first.text); i+=1
            while i<len(selected) and len(group)<config.max_blocks:
                candidate=selected[i]
                # keep structural companions in the same durable source boundary
                same_boundary = candidate.ref.document_id==first.ref.document_id and candidate.ref.section==first.ref.section
                structural = {str(first.content_type).upper(), str(candidate.content_type).upper()} <= {"TABLE", "FIGURE_CAPTION"} or {str(first.content_type).upper(), str(candidate.content_type).upper()} <= {"EQUATION", "DEFINITION"}
                same = same_boundary or structural
                if not same and group: break
                if chars+2+len(candidate.text)>config.max_chars or _tokens("\n\n".join(x.text for x in group+[candidate]))>config.max_estimated_tokens: break
                group.append(candidate); chars += 2+len(candidate.text); i+=1
            if chars>config.max_chars or _tokens("\n\n".join(x.text for x in group))>config.max_estimated_tokens:
                diagnostics.append({"code":"SOURCE_PACKET_TOO_LARGE","chunk_id":first.ref.chunk_id})
                continue
            packet_id=hashlib.sha256(canonical_hash(tuple(x.ref for x in group)).encode()).hexdigest()[:24]
            packets.append(SourcePacket(packet_id,tuple(group)))
        fp=canonical_hash({"generation_id":source.generation_id,"topics":terms,"config":config,"packets":[p.packet_id for p in packets]})
        return SourcePlan(source.generation_id,tuple(packets),fp,tuple(diagnostics),getattr(source, "source_fingerprints", {}))

def _tokens(text: str)->int: return len(re.findall(r"\S+", text))
