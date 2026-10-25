from pathlib import Path

from strategy.context_loader import ContextLoader, normalize_document, read_document
from tests.helpers import config


def test_retrieve_prefers_the_relevant_section(tmp_path: Path):
    docs = tmp_path / "notes"
    docs.mkdir()
    (docs / "btc.md").write_text(
        "# BTC risk\n\n"
        + " ".join(["BTC stop loss risk entry momentum"] * 30)
        + "\n\n# Cooking\n\n"
        + " ".join(["soup recipe tomato basil kitchen"] * 30),
        encoding="utf-8",
    )
    cfg = config()
    cfg.strategy.directory = str(docs)
    cfg.strategy.mode = "retrieve"
    cfg.strategy.chunk_chars = 400
    cfg.strategy.top_k = 1
    loaded = ContextLoader(cfg, tmp_path).load("BTC stop loss risk")
    assert loaded.mode == "retrieve"
    assert "BTC" in loaded.text
    assert "tomato" not in loaded.text


def test_full_mode_includes_every_file(tmp_path: Path):
    docs = tmp_path / "notes"
    docs.mkdir()
    (docs / "a.md").write_text("# Alpha\n\nKeep the stop.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Second note about ETH.\n", encoding="utf-8")
    cfg = config()
    cfg.strategy.directory = str(docs)
    cfg.strategy.mode = "full"
    cfg.strategy.full_max_chars = 50_000
    loaded = ContextLoader(cfg, tmp_path).load("ETH")
    assert loaded.mode == "full"
    assert "Alpha" in loaded.text
    assert "Second note" in loaded.text
    assert loaded.sources == ["a.md", "b.txt"]


def test_pdf_word_per_line_text_is_joined_and_the_knowledge_base_loads():
    assert normalize_document("entry\nstop\nloss\nrules\n") == "entry stop loss rules"
    root = Path(__file__).resolve().parents[1]
    text = read_document(root / "strategy_context" / "Crypto Trading Agent Knowledge Base.pdf")
    assert "MOM-03" in text
    assert "TREND-01" in text
    cfg = config()
    loaded = ContextLoader(cfg, root).load("MOM-03 MACD RSI stop")
    assert "MOM-03" in loaded.text
    assert loaded.chunk_count >= 1
