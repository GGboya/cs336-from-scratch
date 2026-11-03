"""UTF-8 byte encoding reference implementation using plain loops."""


def text_to_byte_ids(text: str) -> list[int]:
    encoded = text.encode("utf-8")
    byte_ids = []
    for byte in encoded:
        byte_ids.append(byte)
    return byte_ids


def byte_ids_to_text(byte_ids: list[int]) -> str:
    data = bytes(byte_ids)
    return data.decode("utf-8", errors="replace")
