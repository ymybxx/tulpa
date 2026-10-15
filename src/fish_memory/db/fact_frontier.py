def fact_frontier_ctes(*, owner_parameter: str) -> str:
    """Shared consumable message prefix for Fact and snapshot scheduling.

    ``owner_parameter`` is a bind placeholder supplied by repository code.
    Background delivery may wait indefinitely for playback; snapshots must wait
    for extraction of the prefix before that delivery, never for the unavailable
    suffix. A session with no consumable prefix has target zero.
    """
    return f"""
        latest AS (
            SELECT
                session.id AS session_id,
                session.fish_id,
                max(message.sequence) AS latest_sequence,
                min(message.sequence) FILTER (
                    WHERE message.fact_extraction_status = 'waiting_consumption'
                ) AS waiting_sequence
            FROM fish_sessions AS session
            JOIN fishes AS fish ON fish.id = session.fish_id
            JOIN messages AS message ON message.session_id = session.id
            WHERE fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
              AND fish.background_owner IS NOT DISTINCT FROM {owner_parameter}
            GROUP BY session.id, session.fish_id
        ),
        bounds AS (
            SELECT
                latest.session_id,
                latest.fish_id,
                COALESCE(max(message.sequence), 0) AS target_sequence
            FROM latest
            LEFT JOIN messages AS message
                ON message.session_id = latest.session_id
               AND message.sequence < latest.latest_sequence
               AND (
                   latest.waiting_sequence IS NULL
                   OR message.sequence < latest.waiting_sequence
               )
            GROUP BY latest.session_id, latest.fish_id
        )
    """
