import re


RULES = [
    # Maintenance / reliability domain terms.
    (r"\b[Тт]аире\b", "ТОиРе"),
    (r"\b[Тт]аиром\b", "ТОиРом"),
    (r"\b[Тт]аиры\b", "ТОиРы"),
    (r"\b[Тт]аира\b", "ТОиРа"),
    (r"\b[Тт]аиру\b", "ТОиРу"),
    (r"\b[Тт]аир\b", "ТОиР"),
    (r"1С\s+(?:Tair|Tire)", "1С:ТОиР"),
    (r"\b[Rr]isk-менеджмент\b", "риск-менеджмент"),
    (
        r"\bанализ ACA\.\s*Asset Criticality Analysis\.\s*Анализ критичности\b",
        "анализ АСА (Asset Criticality Analysis), анализ критичности",
    ),
    (r"\bанализ IC\b", "анализ АСА"),
    (r"\bACA\b", "АСА"),
    (r"Assed Criticality,\s*analsis", "Asset Criticality Analysis"),
    (r"\bAset\b", "Asset"),
    (r"Relibility,\s*Centret Maintings", "Reliability Centered Maintenance"),
    (r"ориентированное на надёжности", "ориентированное на надёжность"),
    (r"\b[Фф]мека\b", "FMECA"),
    (r"Failor Mot Effect Anolises", "Failure Mode and Effects Analysis"),
    (r"\b[Рр]ои\b", "ROI"),
    (r"Reverse of Investment", "Return on Investment"),
    (r"Revers of Investment", "Return on Investment"),
    # Industrial automation and systems.
    (r"\bKIA\b", "КИПиА"),
    (r"\bАСУ\s+ТП\b", "АСУТП"),
    (r"\bосуто\b", "АСУТП"),
    (r"\bСУТП\b", "АСУТП"),
    (r"\b[сС]кад[аы]\b", "SCADA"),
    (r"\b[эЭ]скад[аы]\b", "SCADA"),
    # Recurrent transcription slips.
    (r"\bYek\b", "Y"),
    (r"\bбуллевы\b", "булевы"),
    (r"\bчеловеко часов\b", "человеко-часов"),
    (r"\bДинамирическим\b", "динамометрическим"),
    (r"\bпродаёт нефть\b", "подаёт нефть"),
    (r"\bстриххолдеры\b", "стейкхолдеры"),
    (r"Ноль\.\s*12 MLN\.ru", "0,12 млн руб."),
    (r"\b12 MLN\.ru\b", "0,12 млн руб."),
    (r"\b[Сс]о скоростью не менее 800 л/\s*То есть", "Со скоростью не менее 800 л/мин. То есть"),
    (r"(?<=\. )со скоростью не менее 800 л/мин", "Со скоростью не менее 800 л/мин"),
    (r"800 л\.\s*В минуту", "800 л/мин"),
    (r"резервуар вертикально-стальной", "резервуар вертикальный стальной"),
    (r"резервуар вертикально стальной", "резервуар вертикальный стальной"),
]


def apply_glossary(text: str) -> str:
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    return text
