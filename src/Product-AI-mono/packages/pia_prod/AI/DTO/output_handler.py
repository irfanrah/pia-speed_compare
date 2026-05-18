import os
from pia_prod.AI.global_config import (
    USER_PARAM_KEY,
    CAMERA_ID_KEY,
    INCIDENT_THRESHOLD_SECOND_KEY,
    INCIDENT_TIMEOUT_SECOND_KEY,
    ORGANIZATION_KEY,
    ROI_NAME_KEY,
    PYTHON_AI_SOURCE_HANDLING_NAME,
    ENV_KEY,
    PROD_KEY,
    UUID_KEY,
)

from pia_prod.AI.utils.init import alarm_producer
from pia_prod.AI.validate.calc_score import calc_score
from pia_prod.AI.utils.utils import get_event_type, get_queue_name


def match_outputs(thumbnail, is_start, user_param, frame_cnt, category_name=None, event_uuid=None):
    if os.getenv(ENV_KEY, "") == PROD_KEY:
        """
        RabbitMQ producer를 이용한 코드
        """
        send_alarm_to_mq(thumbnail, category_name, is_start, user_param, event_uuid)
    else:
        """
        개발용 코드
        """
        calc_score(category_name, is_start, user_param)


def send_alarm_to_mq(thumbnail, category_name, is_start, user_param, event_uuid):
    event_type = get_event_type(user_param)
    queue_name = get_queue_name(event_type)
    mq_message = make_alarm_message(
        user_param, thumbnail, is_start, category_name, event_type, event_uuid
    )
    alarm_producer.result_queue.put(
        [
            mq_message,
            PYTHON_AI_SOURCE_HANDLING_NAME,
            queue_name,
        ]
    )


def make_alarm_message(user_param, thumbnail, is_start, category_name, event_type, event_uuid):
    return {
        CAMERA_ID_KEY: user_param[USER_PARAM_KEY][CAMERA_ID_KEY],
        "type": event_type,
        ORGANIZATION_KEY: user_param[USER_PARAM_KEY][ORGANIZATION_KEY],
        ROI_NAME_KEY: user_param[USER_PARAM_KEY][event_type][category_name][ROI_NAME_KEY],
        "isStart": is_start,
        "thumbnail": thumbnail,
        INCIDENT_THRESHOLD_SECOND_KEY: user_param[USER_PARAM_KEY][event_type][category_name][
            INCIDENT_THRESHOLD_SECOND_KEY
        ],
        INCIDENT_TIMEOUT_SECOND_KEY: user_param[USER_PARAM_KEY][event_type][category_name][
            INCIDENT_TIMEOUT_SECOND_KEY
        ],
        UUID_KEY: event_uuid,
    }
