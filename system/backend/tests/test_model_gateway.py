import pytest

from backend.app.model_gateway.base import ModelGateway, ModelNotConfiguredError, ModelSettings


def test_model_check_requires_a_real_llm_configuration():
    gateway = ModelGateway(
        ModelSettings(
            base_url="",
            model_name="",
            embedding_model="",
            reranker_model="",
        )
    )

    with pytest.raises(ModelNotConfiguredError):
        gateway.check_llm_connection()
