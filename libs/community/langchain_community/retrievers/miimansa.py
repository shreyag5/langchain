import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ragatouille import RAGPretrainedModel
from ragatouille.integrations import RAGatouilleLangChainRetriever

from langchain_community.utilities.miimansa import MiimansaUtility


class MiimansaClinicalTextRetriever(RAGPretrainedModel):
    """
    Miimansa clinical text retriever for document search

    Args:
        metadata (dict): Metadata required for our retriever, prepared using utility script
        direct_hit_model: SentenceTransformer model (recommended : 'mixedbread-ai/mxbai-embed-large-v1') used for direct hit
        direct_hit_threshold (float): Normalized cosine similarity score threshold (recommended : 0.91)
        log_direct_hit (bool): Set to True, if you want to log direct hit queries in log_dir
        log_dir (str): Path of logging directory to save direct_hit_logs.csv

    Example:
        RAG = MiimansaClinicalTextRetriever.from_index('./vector_database/colbert/indexes/Colbert-Experimental')
        retriever = RAG.as_langchain_retriever(
                            metadata = metadata,
                            direct_hit_model = direct_hit_model,
                            direct_hit_threshold=0.91,
                            log_direct_hit=True,
                            log_dir='./logs',
                            k=10)
        retriever.get_relevant_documents("What are the primary endpoints of the study?")
    """

    def as_langchain_retriever(
        self,
        metadata,
        direct_hit_model,
        direct_hit_threshold,
        log_direct_hit,
        log_dir,
        **kwargs: Any,
    ) -> BaseRetriever:
        return MiimansaClinicalLangChainRetriever(
            model=self,
            metadata=metadata,
            direct_hit_model=direct_hit_model,
            direct_hit_threshold=direct_hit_threshold,
            log_direct_hit=log_direct_hit,
            log_dir=log_dir,
            **kwargs,
        )


class MiimansaClinicalLangChainRetriever(RAGatouilleLangChainRetriever):
    """LangChain retriever class for Miimansa clinical text retriever"""

    model: Any
    kwargs: dict = {}
    metadata: dict = {}
    direct_hit_model: Any
    direct_hit_threshold: float = 0.91
    log_direct_hit: bool = False
    log_dir: str = "./logs"

    def __init__(
        self,
        metadata,
        direct_hit_model,
        direct_hit_threshold,
        log_direct_hit,
        log_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metadata = metadata
        self.direct_hit_model = direct_hit_model
        self.direct_hit_threshold = direct_hit_threshold
        self.log_direct_hit = log_direct_hit
        self.log_dir = log_dir

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        try:
            direct_hit_flag, direct_hit_qid = MiimansaUtility.is_direct_hit(
                query,
                self.metadata,
                self.direct_hit_model,
                self.direct_hit_threshold,
            )
        except Exception as e:
            raise ValueError(f"Error in is_direct_hit: {e}")
            direct_hit_flag, direct_hit_qid = False, None

        if direct_hit_flag == True and direct_hit_qid is not None:
            try:
                id = self.metadata["qid2id"][direct_hit_qid]
                context = self.metadata["id2context"][id]
            except KeyError as e:
                raise KeyError(f"Metadata key error: {e}")
                return []

            if self.log_direct_hit == True:
                try:
                    data_dict = {
                        "Timestamp": str(time.asctime()),
                        "Query": str(query),
                        "Question": str(self.metadata["qid2ques"][direct_hit_qid]),
                        "QID": str(direct_hit_qid),
                        "Direct Hit Context": str(context),
                        "ID": str(id),
                    }
                    MiimansaUtility.append_to_csv(
                        os.path.join(self.log_dir, "direct_hit_logs.csv"), data_dict
                    )
                except Exception as e:
                    raise ValueError(f"Error logging direct hit: {e}")
                    return []

            return [Document(page_content=context)]
        else:
            try:
                docs = self.model.search(query, **self.kwargs)
                document_ids = [result["document_id"] for result in docs]
                return [
                    Document(page_content=self.metadata["id2context"][document_id])
                    for document_id in document_ids
                ]
            except Exception as e:
                raise ValueError(f"Error while retrieving documents: {e}")
                return []
