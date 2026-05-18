from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.DaeGu_crowd_people.config import PEOPLE_THRESHOLD, ALARM_THRESHOLD


class CPBase(CategoryBase):
    people_threshold: int = PEOPLE_THRESHOLD
    alarm_threshold: int = ALARM_THRESHOLD
