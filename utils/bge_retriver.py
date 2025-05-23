import logging
from typing import Dict, List

from scipy.sparse import csr_array
import numpy as np

from milvus_model.base import BaseEmbeddingFunction
from milvus_model.utils import import_FlagEmbedding, import_datasets
from milvus_model.sparse.utils import stack_sparse_embeddings

import_datasets()
import_FlagEmbedding()

try:
    from FlagEmbedding import BGEM3FlagModel
except AttributeError as e:
    import sys
    if "google.colab" in sys.modules and "ListView" in str(e):
        print("\033[91mIt looks like you're running on Google Colab. Please restart the session to resolve this issue.\033[0m")
        print("\033[91mFor further details, visit: https://github.com/milvus-io/milvus-model/issues/32.\033[0m")
    raise

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BGEM3EmbeddingFunction(BaseEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 16,
        device: str = "",
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.use_fp16 = use_fp16

        if device == "cpu" and use_fp16 is True:
            logger.warning(
                "Using fp16 with CPU can lead to runtime errors such as 'LayerNormKernelImpl', It's recommended to set 'use_fp16 = False' when using cpu. "
            )

        _model_config = dict(
            {
                "model_name_or_path": model_name,
                "device": device,
                "normalize_embeddings": normalize_embeddings,
                "use_fp16": use_fp16,
            },
            **kwargs,
        )
        _encode_config = {
            "batch_size": batch_size,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
            "return_colbert_vecs": return_colbert_vecs,
        }
        self._model_config = _model_config
        self._encode_config = _encode_config

        self.model = BGEM3FlagModel(**self._model_config)

    def __call__(self, texts: List[str]) -> Dict:
        return self._encode(texts)

    @property
    def dim(self) -> Dict:
        return {
            "dense": self.model.model.model.config.hidden_size,
            "colbert_vecs": self.model.model.colbert_linear.out_features,
            "sparse": len(self.model.tokenizer),
        }

    def _encode(self, texts: List[str]) -> Dict:
        output = self.model.encode(queries=texts, **self._encode_config)
        results = {}
        if self._encode_config["return_dense"] is True:
            results["dense"] = list(output["dense_vecs"])
        if self._encode_config["return_sparse"] is True:
            sparse_dim = self.dim["sparse"]
            results["sparse"] = []
            for sparse_vec in output["lexical_weights"]:
                indices = [int(k) for k in sparse_vec]
                values = np.array(list(sparse_vec.values()), dtype=np.float64)
                row_indices = [0] * len(indices)
                csr = csr_array((values, (row_indices, indices)), shape=(1, sparse_dim))
                results["sparse"].append(csr)
            results["sparse"] =  stack_sparse_embeddings(results["sparse"]).tocsr()
        if self._encode_config["return_colbert_vecs"] is True:
            results["colbert_vecs"] = output["colbert_vecs"]
        return results

    def encode_queries(self, queries: List[str]) -> Dict:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> Dict:
        return self._encode(documents)
