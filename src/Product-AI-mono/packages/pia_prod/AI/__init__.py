# packages/pia_prod/AI/__init__.py

from __future__ import annotations

from importlib import import_module
import re
from typing import TYPE_CHECKING

# === 1. pia_prod.AI 에서 직접 꺼내 쓸 Service 들만 노출 ===
__all__ = [
    "FallService",
    "HarnessService",
    "CPService",
    "IntrusionService",
    "InternVL3TrtFalldownService",
    "InternVL3TrtViolenceService",
    "InternVL3TrtLlmService",
    "HelmetService",
    "WeaponService",
    "PVService",
    "PEService",
    "FTPEService",
    "Qwen3VLEService",
    "Qwen3VLETrtService",
    "PeVle2StageSyncService",
    "PeVle2StageAsyncService",
    "FireService",
    "InternVL3Service",
    "VehicleReverseService",
    "FalldownService",
    "SmokeService",
    "TailgateService",
    "WalljumpService",
    "ESFalldownService",
    "PinchService",
    "PatientService",
    "LoiteringService",
    "PeoplecountingService",
    "PeVqa2StageService",
    "SoilQwen35Service",
    "PEBinaryService",
    "HarnessWithClsService",
    "WorkinghighService",
    "PEDistributionService",
]

_SERVICE_NAMES = set(__all__)


def _camel_to_snake(name: str) -> str:
    """
    CamelCase -> snake_case 변환

    예)
      Fall          -> fall
      VehicleReverse -> vehicle_reverse
      InternVL3TrtFalldown -> intern_vl3_trt_falldown
    """
    s1 = re.sub(r"(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# 기본 규칙: <Name>Service -> modules.<snake_name>.service.<Name>Service
# 예: WeaponService -> modules.weapon.service.WeaponService
_SERVICE_MODULE_OVERRIDE = {
    "FallService": "ax_fall",
    "HarnessService": "ax_harness",
    "CPService": "DaeGu_crowd_people",
    "IntrusionService": "Daegu_intrusion",
    "InternVL3TrtFalldownService": "gangnam_falldown",
    "InternVL3TrtViolenceService": "gangnam_violence",
    "InternVL3TrtLlmService": "vqa_internvl",
    "HelmetService": "khonkaen_helmet",
    "WeaponService": "khonkaen_weapon",
    "PVService": "pe_violence",
    "PEService": "perception_encoder",
    "FTPEService": "ft_pe",
    "Qwen3VLEService": "qwen3_vl_embedding",
    "Qwen3VLETrtService": "qwen3vle_trt",
    "PeVle2StageSyncService": "pe_vle_2stage_sync",
    "PeVle2StageAsyncService": "pe_vle_2stage_async",
    "FireService": "samsung_fire",
    "InternVL3Service": "samsung_internvl3",
    "VehicleReverseService": "vehicle_reverse",
    "FalldownService": "yeonsei_falldown",
    "SmokeService": "yeonsei_smoke",
    "TailgateService": "yonsei_tailgate",
    "WalljumpService": "yonsei_walljump",
    "ESFalldownService": "hyundai_esfalldown",
    "PinchService": "kumho_pinch",
    "PatientService": "vanguard_patient",
    "LoiteringService": "aramco_loitering",
    "PeoplecountingService": "khonkaen_peoplecounting",
    "PeVqa2StageService": "pe_vqa_2stage",
    "SoilQwen35Service": "soil_qwen35",
    "PEBinaryService": "pe_binary",
    "HarnessWithClsService": "glenc_harness",
    "WorkinghighService": "glenc_workinghigh",
    "PEDistributionService": "pe_distribution",
}

# === 2. 정적 분석기(Pylance, mypy 등)를 위한 import (런타임에는 실행 X) ===
if TYPE_CHECKING:
    from .modules.ax_fall.service import FallService
    from .modules.ax_harness.service import HarnessService
    from .modules.DaeGu_crowd_people.service import CPService
    from .modules.Daegu_intrusion.service import IntrusionService
    from .modules.gangnam_falldown.service import InternVL3TrtFalldownService
    from .modules.gangnam_violence.service import InternVL3TrtViolenceService
    from .modules.vqa_internvl.service import InternVL3TrtLlmService
    from .modules.khonkaen_helmet.service import HelmetService
    from .modules.khonkaen_weapon.service import WeaponService
    from .modules.pe_violence.service import PVService
    from .modules.perception_encoder.service import PEService
    from .modules.ft_pe.service import FTPEService
    from .modules.qwen3_vl_embedding.service import Qwen3VLEService
    from .modules.qwen3vle_trt.service import Qwen3VLETrtService
    from .modules.samsung_fire.service import FireService
    from .modules.samsung_internvl3.service import InternVL3Service
    from .modules.vehicle_reverse.service import VehicleReverseService
    from .modules.yeonsei_falldown.service import FalldownService
    from .modules.yeonsei_smoke.service import SmokeService
    from .modules.yonsei_tailgate.service import TailgateService
    from .modules.yonsei_walljump.service import WalljumpService
    from .modules.hyundai_esfalldown.service import ESFalldownService
    from .modules.kumho_pinch.service import PinchService
    from .modules.vanguard_patient.service import PatientService
    from .modules.aramco_loitering.service import LoiteringService
    from .modules.khonkaen_peoplecounting.service import PeoplecountingService
    from .modules.pe_vqa_2stage.service import PeVqa2StageService
    from .modules.pe_vle_2stage_sync.service import PeVle2StageSyncService
    from .modules.pe_vle_2stage_async.service import PeVle2StageAsyncService
    from .modules.soil_qwen35.service import SoilQwen35Service
    from .modules.pe_binary.service import PEBinaryService
    from .modules.glenc_harness.service import HarnessWithClsService
    from .modules.glenc_workinghigh.service import WorkinghighService
    from .modules.pe_distribution.service import PEDistributionService


def __getattr__(name: str):
    """
    from pia_prod.AI import WeaponService

    이렇게 불릴 때만 실제 service 모듈을 import 하는 lazy import.
    """
    if name not in _SERVICE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # "WeaponService" -> "Weapon"
    base = name[: -len("Service")]

    # override 가 있으면 우선 사용, 없으면 규칙대로 camel->snake
    module_name = _SERVICE_MODULE_OVERRIDE.get(name, _camel_to_snake(base))

    try:
        module = import_module(f"{__name__}.modules.{module_name}.service")
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Could not import {name!r}: expected module "
            f"'pia_prod.AI.modules.{module_name}.service'"
        ) from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ImportError(
            f"{name!r} not found in module " f"'pia_prod.AI.modules.{module_name}.service'"
        ) from e
