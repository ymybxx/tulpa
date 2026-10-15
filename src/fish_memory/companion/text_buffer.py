"""Shared punctuation boundaries for streamed speech text."""

TTS_TEXT_MIN_CHARS = 15
TTS_TEXT_BOUNDARIES = frozenset("。！？；，、,.!?;:：\n~～…")
_PAUSE_CLOSERS = frozenset("\"')]})）】》」』”’〉")


class TtsTextBuffer:
    """Buffer text, optionally releasing the first spoken prefix at a pause."""

    def __init__(
        self,
        *,
        min_chars: int = TTS_TEXT_MIN_CHARS,
        boundaries: frozenset[str] = TTS_TEXT_BOUNDARIES,
        first_chunk_at_pause: bool = False,
    ) -> None:
        if min_chars <= 0:
            raise ValueError("min_chars must be positive")
        self.min_chars = min_chars
        self.boundaries = boundaries
        self.first_chunk_at_pause = first_chunk_at_pause
        self._pending = ""
        self._received_offset = 0
        self._sent_offset = 0
        self._first_pause_flush_pending = False

    def append(self, text: str) -> list[str]:
        if not text:
            return []
        self._pending += text
        self._received_offset += len(text)
        chunks: list[str] = []
        while True:
            boundary_end = self._find_eligible_boundary()
            if boundary_end is None:
                return chunks
            chunks.append(self._take(boundary_end, at_pause=True))

    def flush_through(self, end_offset: int) -> list[str]:
        if isinstance(end_offset, bool) or not isinstance(end_offset, int):
            return []
        target = min(max(end_offset, self._sent_offset), self._received_offset)
        length = target - self._sent_offset
        if length <= 0:
            return []
        return [self._take(length)]

    def finish(self) -> list[str]:
        if not self._pending:
            return []
        return [self._take(len(self._pending))]

    def adopt_sent_prefix(self, content: str, sent_offset: int) -> list[str]:
        """Adopt a verified public snapshot without resending its TTS prefix."""
        if not 0 <= sent_offset <= len(content):
            raise ValueError("sent_offset must be within snapshot content")
        self._pending = content[sent_offset:]
        self._received_offset = len(content)
        self._sent_offset = sent_offset
        self._first_pause_flush_pending = False
        chunks: list[str] = []
        while True:
            boundary_end = self._find_eligible_boundary()
            if boundary_end is None:
                return chunks
            chunks.append(self._take(boundary_end, at_pause=True))

    def take_first_pause_flush(self) -> bool:
        """Whether the first chunk in this batch needs a provider commit."""
        pending = self._first_pause_flush_pending
        self._first_pause_flush_pending = False
        return pending

    def reset(self) -> None:
        self._pending = ""
        self._received_offset = 0
        self._sent_offset = 0
        self._first_pause_flush_pending = False

    def _take(self, length: int, *, at_pause: bool = False) -> str:
        if at_pause and self.first_chunk_at_pause and self._sent_offset == 0:
            self._first_pause_flush_pending = True
        chunk = self._pending[:length]
        self._pending = self._pending[length:]
        self._sent_offset += len(chunk)
        return chunk

    def _find_eligible_boundary(self) -> int | None:
        first_pause = self.first_chunk_at_pause and self._sent_offset == 0
        has_spoken_text = False
        for index, character in enumerate(self._pending):
            has_spoken_text = has_spoken_text or character.isalnum()
            if character not in self.boundaries:
                continue
            if first_pause:
                if has_spoken_text and self._is_first_pause(index):
                    # Keep a punctuation run and its closing quote together.
                    # One following spoken character confirms the boundary;
                    # otherwise finish/semantic flush releases the whole tail.
                    # Committing before this lookahead can leave a quote-only
                    # provider task, which DashScope rejects as invalid text.
                    end = index + 1
                    while end < len(self._pending) and (
                        self._pending[end] in self.boundaries
                        or self._pending[end] in _PAUSE_CLOSERS
                        or self._pending[end].isspace()
                    ):
                        end += 1
                    if any(char.isalnum() for char in self._pending[end:]):
                        return end
                    return None
            elif index + 1 > self.min_chars:
                return index + 1
        return None

    def _is_first_pause(self, index: int) -> bool:
        character = self._pending[index]
        before = self._pending[index - 1 : index] if index else ""
        after = self._pending[index + 1 : index + 2]
        # ASCII punctuation can belong to a number, time or address. Wait for
        # one following character when ambiguous; finish/segment flushes still
        # release complete text immediately. Chinese punctuation is unambiguous.
        if character in ".,:" and before.isdigit() and (not after or after.isdigit()):
            return False
        if character == ".":
            return (
                bool(after)
                and after != "."
                and not (
                    before.isascii()
                    and before.isalnum()
                    and after.isascii()
                    and after.isalnum()
                )
            )
        if character == ":":
            return bool(after) and after != "/"
        return True
