from pydantic import BaseModel, field_validator, Field
from typing import List, Optional
from pia_prod.AI.utils.init import logger


class ROIModel(BaseModel):
    """
    기본 ROI는 한 카테고리당 하나이나
    나중에 여러개가 필요하거나
    여러 종류의 ROI 영역이 필요하면 해당 클래스를 상속받아서 개발 필요 ( 개발팀과 협의 필요 )
    """

    roiId: Optional[int] = 0
    polygonCoordinates: Optional[List[int]] = []
    divideCoordinates: Optional[List[int]] = []


class CategoryBase(BaseModel):
    name: str
    incidentThresholdSecond: int = 0
    incidentTimeoutSecond: int = 0
    roi: Optional[ROIModel] = Field(default_factory=lambda: ROIModel())

    @field_validator("roi", mode="before")
    @classmethod
    def validate_roi(cls, v):
        if v is None:
            return ROIModel()

        # roi = ["x1", "y1", "x2", "y2", ...] 형태로 들어올 경우 polygonCoordinates에 넣기
        if isinstance(v, list) and all(isinstance(i, (int, float)) for i in v):
            return ROIModel(polygonCoordinates=v)

        # roi가 여러개 들어오는 경우엔 첫번째 것만 사용
        if isinstance(v, list):
            first = v[0] if v else {}
            return ROIModel(**first) if isinstance(first, dict) else first

        # ROI 단일 dict 로 들어오는 경우 mapping
        if isinstance(v, dict):
            return ROIModel(**v)

        # 이미 ROIModel이라면 바로 반환
        if isinstance(v, ROIModel):
            return v

        logger.warning(f"Unexpected roi type: {type(v)}. Returning default ROIModel.")
        return ROIModel()


class RetrievalBase(CategoryBase):
    abnormalText: List[str]
    normalText: List[str]
    topCandidates: int


class VQABase(CategoryBase):
    pass
