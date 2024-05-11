import importlib
from typing import Dict, Type


def get_qualified_name(cls: Type) -> str:
    return f'{cls.__module__}:{cls.__qualname__}'


def ensure_qualified_name(class_name: str, module_name: str) -> str:
    if ':' not in class_name:
        class_name = f'{module_name}:{class_name}'
    return class_name


def import_class(class_name: str):
    if ':' not in class_name:
        raise ValueError(f"Cannot import class {class_name}, use ':' to separate module and class name")
    module_name, class_name = class_name.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate_class(class_name: str, param: Dict):
    return import_class(class_name)(**param)
