import asyncpg


async def create_pool(database_url: str, *, desktop_mode: bool = False) -> asyncpg.Pool:
    if desktop_mode:
        # A desktop instance rarely needs ten eager connections. Start one and
        # grow only when concurrent work actually needs it.
        return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=4)
    return await asyncpg.create_pool(dsn=database_url)
