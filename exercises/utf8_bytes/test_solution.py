from exercises.test_utils import check_case
from exercises.utf8_bytes.solution import byte_ids_to_text, text_to_byte_ids


def test_ascii():
    check_case("ASCII 字符各对应一个字节", text_to_byte_ids, ("A!",), [65, 33])


def test_chinese():
    check_case("中文字符编码为多个字节", text_to_byte_ids, ("你",), [228, 189, 160])


def test_emoji():
    check_case("emoji 编码为四个字节", text_to_byte_ids, ("🙂",), [240, 159, 153, 130])


def test_empty_text():
    check_case("空字符串", text_to_byte_ids, ("",), [])


def test_mixed_text_keeps_byte_order():
    check_case(
        "混合文本保持字节顺序",
        text_to_byte_ids,
        ("A中🙂",),
        [65, 228, 184, 173, 240, 159, 153, 130],
    )


def test_decode_valid_utf8():
    check_case("UTF-8 字节解码", byte_ids_to_text, ([65, 228, 189, 160],), "A你")


def test_decode_invalid_byte_uses_replacement_character():
    check_case("非法字节使用 U+FFFD", byte_ids_to_text, ([65, 255, 66],), "A�B")


def test_decode_truncated_utf8_uses_replacement_character():
    check_case("不完整 UTF-8 序列不抛异常", byte_ids_to_text, ([228, 189],), "�")


def test_decode_empty_bytes():
    check_case("空字节序列解码", byte_ids_to_text, ([],), "")
