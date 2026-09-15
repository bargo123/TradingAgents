"""Stable source-grouped train/validation/test assignment."""
from __future__ import annotations
import hashlib
from typing import Any

class GroupedSplitter:
    def __init__(self, *, ratios=(.70, .15, .15), min_groups=3): self.ratios=ratios; self.min_groups=min_groups
    def assign(self, examples: list[Any]) -> dict[str, str]:
        groups={self._group(e) for e in examples}
        if len(groups) < self.min_groups: raise ValueError("INSUFFICIENT_DATA")
        ordered=sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        n=len(ordered); cut1=max(1, round(n*self.ratios[0])); cut2=max(cut1+1, round(n*(self.ratios[0]+self.ratios[1])))
        return {g:("train" if i<cut1 else "validation" if i<cut2 else "test") for i,g in enumerate(ordered)}
    def _group(self,e):
        if isinstance(e,dict): return str(e.get("document_id", ""))+"|"+str(e.get("chapter", ""))+"|"+str(e.get("section", e.get("section_path", "")))
        return str(getattr(e,"document_id",""))+"|"+str(getattr(e,"chapter",""))+"|"+str(getattr(e,"section",getattr(e,"section_path","")))

def validate_no_group_leakage(examples: list[Any], assignments: dict[str,str]) -> bool:
    seen={}
    for e in examples:
        g=GroupedSplitter()._group(e); split=assignments.get(g)
        if g in seen and seen[g]!=split: return False
        seen[g]=split
    return True
