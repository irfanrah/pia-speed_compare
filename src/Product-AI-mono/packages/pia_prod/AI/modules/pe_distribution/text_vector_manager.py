from collections import defaultdict
import torch
from pia_prod.AI.global_config import VLM_EVENT_KEY, USER_PARAM_KEY


class TxtvecManager:
    def __init__(self, model, device):
        """desc."""
        self.model = model
        self.device = device
        self.txt_vec_dict = defaultdict(  # dict[stream_id][cat_id]
            lambda: defaultdict(lambda: defaultdict(torch.tensor))
        )
        # self.txt_manager_dict = defaultdict(lambda: defaultdict(dict))

    def get_txt_vector(self, prompts):
        return self.model(text=prompts)

    def get_txt_vector_group_by_stream_id(self, stream_id):
        return self.txt_vec_dict[stream_id]

    def get_txt_vector_group_by_category(self, stream_id, category_id):
        stream_txtvec = self.get_txt_vector_group_by_stream_id(stream_id)
        return stream_txtvec[category_id]

    def get_txt_vector_group_by_category_values(self, stream_id, category_id):
        return torch.stack(
            list(self.get_txt_vector_group_by_category(stream_id, category_id).values()), dim=0
        ).squeeze(dim=1)

    def delete_non_used_txtvec(self, stream_idx, cat_idx, all_prompts):
        for prompt in self.txt_vec_dict[stream_idx][cat_idx]:
            if prompt not in all_prompts:
                del self.txt_vec_dict[stream_idx][cat_idx][prompt]

    def update(self, stream_ids, user_params) -> torch.tensor:
        """self.txt_vec_dict에 없는 프롬프트에 대한 벡터를 구한 뒤 딕셔너리에 저장합니다."""
        target_txt_vector = torch.tensor([], device=self.device)
        target_sentences = []
        for stream_id, user_param in zip(stream_ids, user_params):
            for event in user_param[USER_PARAM_KEY][VLM_EVENT_KEY]:
                cat_id = event.name
                abnormal_texts = event.abnormalText
                normal_texts = event.normalText
                all_prompts = abnormal_texts + normal_texts
                for prompt in all_prompts:
                    if prompt not in self.txt_vec_dict[stream_id][cat_id]:
                        unormed_txt_vector = self.get_txt_vector(prompt)
                        self.txt_vec_dict[stream_id][cat_id][prompt] = (
                            unormed_txt_vector / unormed_txt_vector.norm(dim=-1)
                        )
                    target_txt_vector = torch.cat(
                        (target_txt_vector, self.txt_vec_dict[stream_id][cat_id][prompt]), dim=0
                    )
                    target_sentences.append(prompt)
                self.delete_non_used_txtvec(stream_id, cat_id, all_prompts)
        return target_txt_vector, target_sentences
