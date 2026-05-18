from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.khonkaen_peoplecounting.config import PEOPLE_THRESHOLD


class PeoplecountingModel(CategoryBase):
    people_threshold: int = PEOPLE_THRESHOLD
