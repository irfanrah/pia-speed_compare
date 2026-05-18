from typing import List, Optional, Union

from pia_prod.AI.DTO.param_base import RetrievalBase, VQABase
from pia_prod.AI.modules.DaeGu_crowd_people.param import CPBase
from pia_prod.AI.modules.Daegu_intrusion.param import IntrusionModel
from pia_prod.AI.modules.ax_fall.param import FallModel
from pia_prod.AI.modules.samsung_fire.param import FireModel
from pia_prod.AI.modules.yeonsei_falldown.param import FalldownModel
from pia_prod.AI.modules.yeonsei_smoke.param import SmokeModel
from pia_prod.AI.modules.yonsei_walljump.param import WalljumpModel
from pia_prod.AI.modules.yonsei_tailgate.param import TailgateModel
from pia_prod.AI.modules.khonkaen_helmet.param import HelmetModel
from pia_prod.AI.modules.khonkaen_weapon.param import WeaponModel
from pia_prod.AI.modules.ax_harness.param import HarnessModel
from pia_prod.AI.modules.vehicle_reverse.param import VehicleReverseModel
from pia_prod.AI.modules.kumho_pinch.param import PinchModel
from pia_prod.AI.modules.vanguard_patient.param import PatientModel
from pia_prod.AI.modules.aramco_loitering.param import LoiteringModel
from pia_prod.AI.modules.khonkaen_peoplecounting.param import PeoplecountingModel
from pia_prod.AI.modules.glenc_harness.param import HarnessWithClsModel
from pia_prod.AI.modules.glenc_workinghigh.param import WorkinghighModel

from pia_prod.AI.modules.Daegu_intrusion.config import INTRUSION_CV_CATEGORY
from pia_prod.AI.modules.ax_fall.config import FALL_CV_CATEGORY
from pia_prod.AI.modules.ax_harness.config import HARNESS_CV_CATEGORY
from pia_prod.AI.modules.yeonsei_falldown.config import FALLDOWN_CV_CATEGORY
from pia_prod.AI.modules.samsung_fire.config import FIRE_CV_CATEGORY
from pia_prod.AI.modules.DaeGu_crowd_people.config import CROWDED_CV_CATEGORY
from pia_prod.AI.modules.yeonsei_smoke.config import SMOKE_CV_CATEGORY
from pia_prod.AI.modules.yonsei_walljump.config import WALLJUMP_CV_CATEGORY
from pia_prod.AI.modules.yonsei_tailgate.config import TAILGATE_CV_CATEGORY
from pia_prod.AI.modules.perception_encoder.config import INTERNVL3_VQA_CATEGORY
from pia_prod.AI.modules.khonkaen_helmet.config import HELMET_CV_CATEGORY
from pia_prod.AI.modules.khonkaen_weapon.config import WEAPON_CV_CATEGORY
from pia_prod.AI.modules.vehicle_reverse.config import VEHICLE_REVERSE_CV_CATEGORY
from pia_prod.AI.modules.kumho_pinch.config import PINCH_CV_CATEGORY
from pia_prod.AI.modules.vanguard_patient.config import PATIENT_CV_CATEGORY
from pia_prod.AI.modules.aramco_loitering.config import LOITERING_CV_CATEGORY
from pia_prod.AI.modules.khonkaen_peoplecounting.config import PEOPLECOUNTING_CV_CATEGORY
from pia_prod.AI.modules.glenc_harness.config import HARNESS_WITH_CLS_CV_CATEGORY
from pia_prod.AI.modules.glenc_workinghigh.config import WORKINGHIGH_CV_CATEGORY

from pydantic import BaseModel, field_validator


class AddStreamModel(BaseModel):  # /api/v1/stream/add
    cameraId: int
    cameraUrl: str
    organization: str
    retEvent: Optional[List[RetrievalBase]] = []
    cvEvent: Optional[
        List[
            Union[
                IntrusionModel,
                FalldownModel,
                FireModel,
                CPBase,
                SmokeModel,
                WalljumpModel,
                TailgateModel,
                HelmetModel,
                WeaponModel,
                FallModel,
                HarnessModel,
                VehicleReverseModel,
                PinchModel,
                PatientModel,
                LoiteringModel,
                PeoplecountingModel,
                HarnessWithClsModel,
                WorkinghighModel,
            ]
        ]
    ] = []
    vqaEvent: Optional[List[VQABase]] = []
    timestamp: str  # time (%Y-%m-%d %H:%M:%S) (YYYY-MM-DDTHH:mm:ss.ffffff) UTC ISO 8601

    @field_validator("cvEvent", mode="before")
    @classmethod
    def parse_cv_events(cls, v):
        if not isinstance(v, list):
            raise ValueError("cvEvent must be a list")

        parsed_event_list = []
        for event in v:
            if not isinstance(event, dict):
                event = dict(event)
            name = event.get("name")

            if name in INTRUSION_CV_CATEGORY:
                parsed_event_list.append(IntrusionModel(**event))
            elif name in FALLDOWN_CV_CATEGORY:
                parsed_event_list.append(FalldownModel(**event))
            elif name in FIRE_CV_CATEGORY:
                parsed_event_list.append(FireModel(**event))
            elif name in CROWDED_CV_CATEGORY:
                parsed_event_list.append(CPBase(**event))
            elif name in SMOKE_CV_CATEGORY:
                parsed_event_list.append(SmokeModel(**event))
            elif name in WALLJUMP_CV_CATEGORY:
                parsed_event_list.append(WalljumpModel(**event))
            elif name in TAILGATE_CV_CATEGORY:
                parsed_event_list.append(TailgateModel(**event))
            elif name in INTERNVL3_VQA_CATEGORY:
                parsed_event_list.append(VQABase(**event))
            elif name in HELMET_CV_CATEGORY:
                parsed_event_list.append(HelmetModel(**event))
            elif name in WEAPON_CV_CATEGORY:
                parsed_event_list.append(WeaponModel(**event))
            elif name in FALL_CV_CATEGORY:
                parsed_event_list.append(FallModel(**event))
            elif name in HARNESS_CV_CATEGORY:
                parsed_event_list.append(HarnessModel(**event))
            elif name in VEHICLE_REVERSE_CV_CATEGORY:
                parsed_event_list.append(VehicleReverseModel(**event))
            elif name in PINCH_CV_CATEGORY:
                parsed_event_list.append(PinchModel(**event))
            elif name in PATIENT_CV_CATEGORY:
                parsed_event_list.append(PatientModel(**event))
            elif name in LOITERING_CV_CATEGORY:
                parsed_event_list.append(LoiteringModel(**event))
            elif name in PEOPLECOUNTING_CV_CATEGORY:
                parsed_event_list.append(PeoplecountingModel(**event))
            elif name in HARNESS_WITH_CLS_CV_CATEGORY:
                parsed_event_list.append(HarnessWithClsModel(**event))
            elif name in WORKINGHIGH_CV_CATEGORY:
                parsed_event_list.append(WorkinghighModel(**event))
            else:
                raise ValueError(f"Unknown category name: {event}")
        return parsed_event_list

    @field_validator("retEvent", mode="before")
    @classmethod
    def parse_ret_events(cls, v):
        if not v:
            return []
        if not isinstance(v, list):
            raise ValueError("retEvent must be a list")

        parsed = []
        for event in v:
            if not isinstance(event, dict):
                event = dict(event)
            parsed.append(RetrievalBase(**event))
        return parsed

    @field_validator("vqaEvent", mode="before")
    @classmethod
    def parse_vqa_events(cls, v):
        if not v:
            return []
        if not isinstance(v, list):
            raise ValueError("vqaEvent must be a list")

        parsed = []
        for event in v:
            if not isinstance(event, dict):
                event = dict(event)
            parsed.append(VQABase(**event))
        return parsed

    class Config:
        extra = "allow"  # 추가 변수 허용


class DeleteStreamModel(BaseModel):  # /api/v1/stream/remove
    cameraId: int
    cameraUrl: str
    organization: str


# python -> backend


class RTSPErrorModel(BaseModel):  # /api/camera/inform/error
    cameraId: int
    organization: str
    timestamp: str  # time (%Y-%m-%d %H:%M:%S)
    state: int


class CreatedConnectionModel:  # api/ai-manager/connect
    """
    AI Manager와의 연결을 설정하는 모델.

    Attributes:
        key (str): AI Manager와 연결하는 인증 키.
    """

    key: str


class LoggingModel(BaseModel):
    logging_state: bool
