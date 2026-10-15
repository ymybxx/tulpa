from fish_memory.facts.embedding import build_fact_embedding_text


def test_embedding_text_uses_fact_semantics_without_role_boilerplate() -> None:
    assert (
        build_fact_embedding_text(
            text="该用户每天打篮球",
            what="每天打篮球",
            where=(),
            who=("该用户",),
            entities=("篮球",),
        )
        == "每天打篮球"
    )


def test_embedding_text_keeps_named_context_and_location() -> None:
    assert (
        build_fact_embedding_text(
            text="该用户在Target兑换了一张优惠券",
            what="兑换一张优惠券",
            where=("Target",),
            who=("该用户",),
            entities=("Target", "优惠券"),
        )
        == "兑换一张优惠券 Target"
    )
