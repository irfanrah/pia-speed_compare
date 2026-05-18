from collections import defaultdict, deque
from typing import Union

import numpy as np
import torch


class TTLQueue:
    def __init__(self, maxlen) -> None:
        self.TTL = 0
        self.q = deque(maxlen=maxlen)


class MultiQueueManager:
    def __init__(self, max_queue_lenth, max_queue_num, max_TTL=30) -> None:
        self.queue_lenth = max_queue_lenth  # temporal size
        self.max_queue_num = max_queue_num  # for memory efficients
        self.max_TTL = max_TTL  # Time To Live - 네트워크에서 주로 사용하는 용어
        self.manager = defaultdict(TTLQueue)

    def check_queue_num(self):
        if len(self.manager) >= self.max_queue_num:
            return False
        return True

    def put(self, index, data):
        # Key가 없으면 생성
        if index not in self.manager.keys():
            self.create_queue(index=index)
            self.fill_queue(index=index, data=data)

        self.manager[index].TTL = 0  # 새로운 데이터가 들어왔으면 TTL을 초기화
        self.manager[index].q.append(data)

    def increase_TTL(self):
        for _, v in self.manager.items():
            v.TTL += 1
        self.delete_old_queue()  # 증가 후 늙은 queue를 삭제

    def create_queue(self, index):
        if not self.check_queue_num():
            # 만약 큐의 개수가 꽉 찻으면 가장 업데이트된지 오래된 큐가 가장 먼저 삭제 된다.
            del_target_index = self.get_highest_TTL()
            self.delete_queue(del_target_index)
        self.manager[index] = TTLQueue(maxlen=self.queue_lenth)

    def delete_old_queue(self):
        delete_targets = []
        for k, v in self.manager.items():
            if v.TTL >= self.max_TTL:
                delete_targets.append(k)

        for target in delete_targets:
            self.manager.pop(target)

    def get_highest_TTL(self):
        max_TTL = -1
        for k, v in self.manager.items():
            if v.TTL > max_TTL:
                highest_TTL = k
            max_TTL = max(v.TTL, max_TTL)
        return highest_TTL

    def delete_queue(self, index):
        try:
            self.manager.pop(index)
            return True
        except Exception:
            return False

    def get_data(self, index, return_type="list") -> Union[list, torch.Tensor, np.ndarray]:
        """
        Param :
            index : Any
                queue index
            return_type : str
                If you set the wanted type, it returns the list in that type.
                supported types = list, np, torch
        """
        if index in self.manager:
            if return_type != "list":
                ret = np.array(self.manager[index].q)
                if return_type == "np" or return_type == "numpy":
                    return ret
                elif return_type == "torch":
                    return torch.Tensor(ret)
            else:
                return list(self.manager[index].q)
        else:
            return []

    def fill_queue(self, index, data):
        # max_len -1 만큼 넣고 put 함수 마지막에 1개 너 추가로 append
        while len(self.manager[index].q) < self.queue_lenth - 1:
            self.manager[index].q.append(data)

    def get_all_datas(self, return_type="np"):
        """
        Param :
            return_type : str
                If you set the wanted type, it returns the list in that type.
                supported types = np, torch
        return :
            keys : List[index]
                queues index
            datas : np or torch data
                queues datas
        """
        datas = []
        keys = []
        for key, ttl_cls in self.manager.items():
            datas.append(np.array(ttl_cls.q))
            keys.append(key)

        if return_type == "np":
            return keys, np.array(datas)
        elif return_type == "torch":
            return keys, torch.from_numpy(np.array(datas))
