from uuid import UUID

from fish_memory.recall.masking import (
    EntityAliases,
    build_mask_plan,
    cosine_similarity,
    entity_signature,
    mask_text,
    signature_hash,
)

A = UUID(int=1)
B = UUID(int=2)


def test_masks_every_real_alias_span_for_one_entity() -> None:
    entity = EntityAliases(A, ("杰洛特", "白狼", "Geralt"))

    assert mask_text("白狼以前使用过银剑", (entity,)) == "以前使用过银剑"


def test_masks_multiple_entities_as_one_signature() -> None:
    entities = (
        EntityAliases(A, ("A",)),
        EntityAliases(B, ("B",)),
    )

    assert mask_text("A xxxx B", entities) == "xxxx"


def test_requires_every_signature_entity_to_exist_in_the_key_text() -> None:
    entities = (
        EntityAliases(A, ("白狼",)),
        EntityAliases(B, ("叶奈法",)),
    )

    assert mask_text("白狼使用过银剑", entities) is None


def test_matches_nfkc_casefolded_aliases_without_masking_substrings() -> None:
    geralt = EntityAliases(A, ("geralt",))
    short_alias = EntityAliases(B, ("he",))

    assert mask_text("ＧＥＲＡＬＴ used a sword", (geralt,)) == "used a sword"
    assert mask_text("the sword", (short_alias,)) is None


def test_signature_is_uuid_sorted_and_has_stable_hash() -> None:
    signature = entity_signature((B, A, B))

    assert signature == (A, B)
    assert signature_hash(signature) == (
        "78d68721debb423cf880232869a63e2522c3dd6187159cbeb15b3ee7e15925f8"
    )


def test_builds_same_signature_query_and_key_masks_from_aliases() -> None:
    entity = EntityAliases(A, ("杰洛特", "白狼"))

    plan = build_mask_plan(
        "白狼以前用过什么剑",
        "杰洛特以前使用过银剑",
        (entity,),
        allowed_entity_ids=(A,),
    )

    assert plan is not None
    assert plan.signature == (A,)
    assert plan.masked_query == "以前用过什么剑"
    assert plan.masked_key == "以前使用过银剑"


def test_fact_mask_plan_rejects_entity_without_structured_link() -> None:
    entity = EntityAliases(A, ("杰洛特", "白狼"))

    assert (
        build_mask_plan(
            "白狼以前用过什么剑",
            "杰洛特以前使用过银剑",
            (entity,),
            allowed_entity_ids=(B,),
        )
        is None
    )


def test_cosine_similarity_uses_exact_vectors() -> None:
    assert cosine_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
