import cv2
import time
from queue import Queue
from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time
import pandas as pd
from overriding_service.perception_encoder_service import ProfilePEService


def get_AddStreamModel_custom() -> tuple:
    def _create_add_stream_model(camera_id: int) -> tuple:
        user_param = {
            "user_param": {
                "cameraId": str(camera_id),
                "cameraUrl": "0",
                "organization": "pia",
                "retEvent": {
                    "falldown_ret": {
                        "name": "falldown_ret",
                        "incidentThresholdSecond": 3,
                        "incidentTimeoutSecond": 3,
                        "confidenceThreshold": 0.5,
                        "nmsThreshold": 0.5,
                        "roi": {
                            "roiId": 1,
                            "polygonCoordinates": [],
                        },
                    },
                    "fire_ret": {
                        "name": "fire_ret",
                        "incidentThresholdSecond": 3,
                        "incidentTimeoutSecond": 3,
                        "confidenceThreshold": 0.5,
                        "nmsThreshold": 0.5,
                        "roi": {
                            "roiId": 1,
                            "polygonCoordinates": [],
                        },
                    },
                    "smoke_ret": {
                        "name": "smoke_ret",
                        "incidentThresholdSecond": 3,
                        "incidentTimeoutSecond": 3,
                        "confidenceThreshold": 0.5,
                        "nmsThreshold": 0.5,
                        "roi": {
                            "roiId": 1,
                            "polygonCoordinates": [],
                        },
                    },
                },
                "timestamp": str_UTC_ISO8601_ms_now_time(),
            }
        }

        stream_id = f"{user_param['user_param']['cameraId']}_{user_param['user_param']['organization']}"
        return user_param, stream_id

    return _create_add_stream_model


def test_pe_batches(get_video=None, max_batch_size=None):
    q = Queue(100)

    service = ProfilePEService(q)

    for j in range(max_batch_size):
        now_batch_size = j + 1
        service.time_dict = {"total": [], "model_preprocess": [], "model_inference": [], "postprocess_logic": [], "send_alarm": []}

        print(f"================================== Start Test Batch Size: {now_batch_size} ==================================")
        cap = cv2.VideoCapture(get_video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        assert cap is not None, "Failed to open video"
        user_params = []
        stream_ids = []
        for i in range(now_batch_size):
            user_param_i, stream_id_i = get_AddStreamModel_custom()(i)
            user_params.append(user_param_i)
            stream_ids.append(stream_id_i)

        count = 0
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(10)  # 큐 다 빠져나갈때까지 안정하게 10초정도 대기
                    average_times = {k: sum(v) / len(v) if len(v) > 0 else 0 for k, v in service.time_dict.items()}
                    df = pd.DataFrame([average_times])
                    print(df.to_string(index=False))
                    print(f"================================== End Test Batch Size: {now_batch_size} ==================================")
                    break

                batches = [frame] * now_batch_size
                assert len(batches) == now_batch_size, f"Batch size should be {now_batch_size}"

                if count % round(fps * OD_TIME_INTERVAL_SECOND) == 0:
                    q.put(
                        {
                            "batches": batches,
                            "stream_ids": stream_ids,
                            "user_params": user_params,
                        }
                    )
                count += 1

    else:
        assert "Failed to open video 1 or 2", "Video capture could not be opened"


if __name__ == "__main__":
    OD_TIME_INTERVAL_SECOND = 0.5
    VIDEO_PATH = "/home/gpuadmin/Downloads/samsung_fire2.mp4"
    MAX_BATCH_SIZE = 16
    test_pe_batches(get_video=VIDEO_PATH, max_batch_size=MAX_BATCH_SIZE)
