import numpy as np
from pia.utils.filters.base import FilterConfigBase
from pia.utils.filters.factory import FilterFactory


def test_filters():
    datas = np.random.random(size=(100))

    for name in FilterFactory.get_avaliable_filters():
        config = FilterConfigBase(
            filter_type=name,
            window_length=3,
            alpha_weight=0.2,
            polyorder=2,
        )
        filter = FilterFactory.load(config)
        for d in datas:
            print(f"{filter.get_name()} : {filter(obs=d)}")


if __name__ == "__main__":
    test_filters()
