from langgraph.graph import MessagesState
from typing import Optional, List, Dict, Literal

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    title: str = ""
    description: str = ""
    status: Literal["pending", "completed"] = "pending"


class Plan(BaseModel):
    goal: str = ""
    thought: str = ""
    steps: List[Step] = []

class State(MessagesState):
    user_message: str = ""
    plan: Plan
    observations: List = []
    final_report: str =  ""
    