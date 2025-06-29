from pydantic import BaseModel, Field
from typing import List


class Match(BaseModel):
    name: str
    net_worth: float
    age: int
    source: str
    similarity_score: float = Field(..., ge=-1.0, le=1.0)


class PredictResponse(BaseModel):
    estimated_net_worth: float  # USD
    top_matches: List[Match] 