from unittest.mock import Mock, patch

from src.search_service import SearchService


def _service() -> SearchService:
    return SearchService(
        searxng_public_instances_enabled=False,
        rag_api_url="http://rag.example/",
        rag_api_key="secret",
    )


def test_rag_news_uses_mandatory_single_ticker_filter() -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "results": [
            {
                "chunk_text": "NVIDIA reports new results",
                "metadata": {
                    "ticker_symbols": ["NVDA"],
                    "source_name": "polygon",
                    "source_url": "https://example.com/nvda",
                    "published_at": "2026-08-26T00:00:00+00:00",
                },
            },
            {
                "chunk_text": "Unrelated Apple news",
                "metadata": {
                    "ticker_symbols": ["AAPL"],
                    "source_url": "https://example.com/aapl",
                },
            },
        ]
    }

    with patch("src.search_service._post_with_retry", return_value=response) as request:
        result = _service()._search_rag_news(
            query="NVDA latest news",
            stock_code="NVDA",
            max_results=5,
            search_days=3,
        )

    payload = request.call_args.kwargs["json"]
    assert payload["tickers"] == ["NVDA"]
    assert payload["data_types"] == [
        "news",
        "earnings",
        "sec_filing",
        "regulatory",
        "insider",
    ]
    assert [item.url for item in result.results] == ["https://example.com/nvda"]
    assert result.results[0].relevance_category == "direct_company_news"


def test_rag_failure_is_fail_open() -> None:
    with patch("src.search_service._post_with_retry", side_effect=TimeoutError("offline")):
        result = _service()._search_rag_news(
            query="NVDA latest news",
            stock_code="NVDA",
            max_results=5,
            search_days=3,
        )

    assert result.success is False
    assert result.results == []
