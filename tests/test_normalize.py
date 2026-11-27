import pytest

from gigaam.utils import normalize_raw_text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello WORLD", "hello world"),
        ("MiXeD CaSe", "mixed case"),
        ("foo   bar\t\nbaz", "foo bar baz"),
        ("  leading and trailing  ", "leading and trailing"),
        ("Hello, world!", "hello world"),
        # punctuation and hyphens split words
        ("из-за дождя", "из за дождя"),
        ("word-internal hyphen", "word internal hyphen"),
        ("a.b,c-d", "a b c d"),
        ("3,14 6:6 1/2", "3 14 6 6 1 2"),
        ("слово,слово...слово", "слово слово слово"),
        ('foo"bar', "foo bar"),
        ("foo''bar", "foo bar"),
        ("...!!!", ""),
        ("дом 9400 кв", "дом 9400 кв"),
        ("Привет, мир!", "привет мир"),
        ("ЁЖИК ёлка", "ежик елка"),
        ("გამარჯობა, მსოფლიო!", "გამარჯობა მსოფლიო"),
        ("ქართული ენა — 2024", "ქართული ენა 2024"),
        ("Привет, world!", "привет world"),
        # word-internal apostrophes are kept
        ("Окей, let's go!", "окей let's go"),
        ("I don't know.", "i don't know"),
        ("We're here, can't stop!", "we're here can't stop"),
        ("O'Brien", "o'brien"),
        ("it’s fine", "it's fine"),  # curly apostrophe -> ASCII
        ("o‘zbek g‘oz", "o'zbek g'oz"),  # U+2018 (Uzbek) -> ASCII
        # standalone quotes / possessive trailing apostrophes are dropped
        ("'quoted' text", "quoted text"),
        ("dogs' toys", "dogs toys"),
        ("Oʻzbekiston — mustaqil davlat!", "o'zbekiston mustaqil davlat"),
        ("Сәлем, әлем!", "сәлем әлем"),
        ("Қазақстан Республикасы", "қазақстан республикасы"),
        ("Салам, дүйнө!", "салам дүйнө"),
        ("Кыргыз Республикасы — 2024", "кыргыз республикасы 2024"),
        # Armenian: internal marks are dropped. Punctuation splits words
        ("Բարև, աշխարհ։", "բարև աշխարհ"),
        ("Հայաստանի Հանրապետություն", "հայաստանի հանրապետություն"),
        ("100 մետրանոց նավը։", "100 մետրանոց նավը"),
        ("Ո՞վ է այնտեղ՝ Արամը։", "ով է այնտեղ արամը"),
        ("Արա՛մ", "արամ"),
        ("ինչպե՞ս", "ինչպես"),
        ("ինչպե՜ս", "ինչպես"),
        ("Բարև։աշխարհ", "բարև աշխարհ"),
        ("մյուս դեպքերում, գրանցվելու՜ կարիք.", "մյուս դեպքերում գրանցվելու կարիք"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_raw_text(text, expected):
    assert normalize_raw_text(text) == expected


@pytest.mark.parametrize(
    "text, language, expected",
    [
        # Azerbaijani I/İ casing differs from the default lowercase
        ("AZƏRBAYCAN I İ ŞƏKİ", "az", "azərbaycan ı i şəki"),
        ("I İ", "en", "i i"),
    ],
)
def test_normalize_language(text, language, expected):
    assert normalize_raw_text(text, language=language) == expected
