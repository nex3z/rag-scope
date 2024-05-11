from abc import ABC

from pydantic import BaseModel, ConfigDict


class BasePresenter(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
