# Now the rest of the imports
# from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.pe import CLIP as PE_CLIP
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.pe as pe
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.transforms as transforms

if __name__ == "__main__":
    device = "cuda"
    model_name = "PE-Core-L14-336"
    model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)

    print("Successfully loaded PE-Core-L14-336 model")
    print("Successfully loaded PE tokenizer")
