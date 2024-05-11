from abc import ABC, abstractmethod


class BaseView(ABC):
    @abstractmethod
    def build(self) -> 'BaseView':
        pass
