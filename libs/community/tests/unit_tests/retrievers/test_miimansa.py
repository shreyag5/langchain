import pytest
from unittest.mock import patch, MagicMock
from langchain_community.retrievers.miimansa import MiimansaClinicalTextRetriever, MiimansaClinicalLangChainRetriever

@pytest.fixture
def mock_metadata():
    return {
        "qid2id": {"qid1": "id1"},
        "id2context": {"id1": "Context_id1"},
        "qid2ques": {"qid1": "Question_qid1"}
    }

@pytest.fixture
def mock_direct_hit_model():
    return MagicMock()

@pytest.fixture
def retriever(mock_metadata, mock_direct_hit_model):
    return MiimansaClinicalTextRetriever()

def test_as_langchain_retriever(retriever, mock_metadata, mock_direct_hit_model):
    langchain_retriever = retriever.as_langchain_retriever(
        metadata=mock_metadata,
        direct_hit_model=mock_direct_hit_model,
        direct_hit_threshold=0.91,
        log_direct_hit=True,
        log_dir='./logs'
    )
    assert isinstance(langchain_retriever, MiimansaClinicalLangChainRetriever)

def test_get_relevant_documents_when_direct_hit(
    mock_metadata, mock_direct_hit_model, mocker
):
    query = "test query"
    retriever = MiimansaClinicalLangChainRetriever(
        metadata=mock_metadata,
        direct_hit_model=mock_direct_hit_model,
        direct_hit_threshold=0.91,
        log_direct_hit=True,
        log_dir='./logs'
    )
    mocked_utility = MagicMock()
    mocked_utility.is_direct_hit.return_value = (True, "qid1")
    mocker.patch(
        'langchain_community.utilities.miimansa.MiimansaUtility',
        return_value=mocked_utility
    )

    result = retriever._get_relevant_documents(query, run_manager=MagicMock())

    assert len(result) == 1
    assert result[0].page_content == "Context_id1"


def test_get_relevant_documents_when_no_direct_hit(
    mock_metadata, mock_direct_hit_model, mocker
):
    query = "test query"
    retriever = MiimansaClinicalLangChainRetriever(
        metadata=mock_metadata,
        direct_hit_model=mock_direct_hit_model,
        direct_hit_threshold=0.91,
        log_direct_hit=True,
        log_dir='./logs'
    )
    mocked_utility = MagicMock()
    mocked_utility.is_direct_hit.return_value = (False, None)
    mocker.patch(
        'langchain_community.utilities.miimansa.MiimansaUtility',
        return_value=mocked_utility
    )

    mocked_model = MagicMock()
    mocked_model.search.return_value = [{"document_id": "id1"},{"document_id": "id2"}]
    retriever.model = mocked_model

    result = retriever._get_relevant_documents(query, run_manager=MagicMock())

    assert len(result)>0
