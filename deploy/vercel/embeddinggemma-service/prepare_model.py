"""Bundle only the text encoder at build time; runtime needs no Hub download."""

from pathlib import Path

from main import MODEL_ID, MODEL_PATH, MODEL_REVISION


def prepare_model():
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_PATH), safe_serialization=True)
    print(f"Text encoder bundled at {Path(MODEL_PATH)} ({MODEL_REVISION})")


if __name__ == "__main__":
    prepare_model()
