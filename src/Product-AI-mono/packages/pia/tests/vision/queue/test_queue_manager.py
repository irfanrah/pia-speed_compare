import numpy as np
import torch

from pia.vision.queue.manager import MultiQueueManager


def test_put_and_retrieve_data():
    manager = MultiQueueManager(max_queue_lenth=3, max_queue_num=3)
    manager.put("first", 0)
    manager.put("first", 1)
    manager.put("second", 5)

    assert manager.get_data("first") == [0, 0, 1]
    assert np.array_equal(manager.get_data("first", return_type="np"), np.array([0, 0, 1]))
    assert torch.allclose(
        manager.get_data("first", return_type="torch"), torch.tensor([0.0, 0.0, 1.0])
    )

    keys, datas = manager.get_all_datas(return_type="np")
    assert set(keys) == {"first", "second"}
    assert datas.shape == (2, 3)


def test_ttl_deletion_removes_stale_queue():
    manager = MultiQueueManager(max_queue_lenth=2, max_queue_num=2, max_TTL=1)
    manager.put("a", 0)
    assert "a" in manager.manager

    manager.increase_TTL()  # TTL becomes 1 and exceeds limit
    assert "a" not in manager.manager
    assert manager.get_data("a") == []
