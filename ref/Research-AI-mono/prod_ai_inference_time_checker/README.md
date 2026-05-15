# prod_ai_inference_time_checker
해당 모듈은 Prod-ai-mono에서 사용하는 카테고리의 소요시간을 체크하기 위해 클래스를 새로 정의하였습니다. 각 스텝별 소요시간(Delay)를 측정할 수 있습니다.

## 예시
```
================================== Start Test Batch Size: 1 ==================================
helmet_cv : True
helmet_cv : False
helmet_cv : True
helmet_cv : False
    total  model1_preprocess  model1_inference  model1_nms  model2_batch_make  model2_preprocess  model2_inference  postprocess_logic  send_alarm
11.591784           3.847823          3.785104    1.768918           0.154147            0.27073          1.449799           0.139332    0.175932
================================== End Test Batch Size: 1 ==================================
================================== Start Test Batch Size: 2 ==================================
helmet_cv : True
helmet_cv : True
helmet_cv : False
helmet_cv : False
helmet_cv : True
helmet_cv : True
helmet_cv : False
helmet_cv : False
    total  model1_preprocess  model1_inference  model1_nms  model2_batch_make  model2_preprocess  model2_inference  postprocess_logic  send_alarm
15.196751           6.399997          3.976147    2.003774           0.270745           0.421876          1.676495           0.170692    0.277025
================================== End Test Batch Size: 2 ==================================

```

## 사용법
1. Prod,Pia-ai-package 등 ai inference 서버전용 세팅을 완료한다.
2. `cd prod_ai_inference_time_checker`
3. `python check_helmet.py > helmet_test.log`
4. `python check_weapon.py > weapon_test.log`
5. `python check_perception_encoder.py > pe_test.log`


## 당부
- 필요하신 카테고리 요청주세요!