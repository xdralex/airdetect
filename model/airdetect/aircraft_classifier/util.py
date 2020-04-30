from typing import Dict


def check_flag(kv: Dict[str, float], flag: str) -> bool:
    return (flag in kv) and kv[flag] >= 0.5
