# -*- coding: utf-8 -*-
"""
Утилита автоматизированного тестирования требований с помощью ИИ.

Рабочий процесс:
  1. Рекурсивно сканирует папку INPUT_DIR (по умолчанию ./input) и все
     вложенные подпапки, читает файлы требований форматов:
     doc, docx, md, pdf, txt, xlsx, xml, json.
  2. Отправляет содержимое файлов + промт из PROMPT_FILE в ИИ
     (OpenAI-совместимый API, модель qwen3.8, temperature=0.1,
     max_tokens=200000). Запрос синхронный (stream=False): утилита
     дожидается ПОЛНОГО ответа модели и только после этого начинает
     обработку результата.
  3. Сохраняет полученный от ИИ результат в файл вида
     "тестирование требований__ГГГГ-ММ-ДД_ЧЧ-ММ-СС.<расширение>"
     в папке OUTPUT_DIR.

Все параметры вынесены в файл .env (см. ENV_KEYS ниже).
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Загружаем настройки из .env в директории проекта (не перекрывая уже
# установленные переменные окружения).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ----------------------------- Настройки из .env ----------------------------

INPUT_DIR = Path(os.getenv("INPUT_DIR", str(BASE_DIR / "input")))
PROMPT_FILE = Path(os.getenv("PROMPT_FILE", str(BASE_DIR / "prompt.txt")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "output")))

# Относительные пути из .env считаем относительно папки проекта.
def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (BASE_DIR / p).resolve()

INPUT_DIR = _resolve(INPUT_DIR)
PROMPT_FILE = _resolve(PROMPT_FILE)
OUTPUT_DIR = _resolve(OUTPUT_DIR)

SUPPORTED_EXTENSIONS = [
    # Приводим к виду ".doc": нормализуем расширение с точкой в начале.
    ("." + e.strip().lower().lstrip("."))
    for e in os.getenv("SUPPORTED_EXTENSIONS", "doc,docx,md,pdf,txt,xlsx,xml,json").split(",")
    if e.strip()
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "qwen3.8")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200000"))
TIMEOUT = float(os.getenv("TIMEOUT", "600"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "5"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — опытный аналитик по тестированию требований.")

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "тестирование требований__")
OUTPUT_EXTENSION = os.getenv("OUTPUT_EXTENSION", ".md")
TIMESTAMP_FORMAT = os.getenv("TIMESTAMP_FORMAT", "%Y-%m-%d_%H-%M-%S")

MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "0"))  # 0 = без ограничения


# ------------------------------ Чтение файлов -------------------------------

def read_txt(path: Path) -> str:
    """Чтение простого текста с подбором кодировки."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    return path.read_text(encoding="utf-8", errors="replace")


def read_docx(path: Path) -> str:
    """Чтение .docx через python-docx, при ошибке — через zip+XML."""
    try:
        import docx  # python-docx

        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception:
        pass
    # Запасной вариант: читаем document.xml напрямую из zip-архива.
    import zipfile

    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return xml.strip()


def read_pdf(path: Path) -> str:
    """Чтение .pdf через pypdf (с запасным вариантом через PyPDF2)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def read_xlsx(path: Path) -> str:
    """Чтение .xlsx через openpyxl со всех листов в виде TSV."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
        lines = []
        for ws in wb.worksheets:
            lines.append(f"=== Лист: {ws.title} ===")
            for row in ws.iter_rows(values_only=True):
                cells = [("" if c is None else str(c)) for c in row]
                if any(cells):
                    lines.append("\t".join(cells))
        wb.close()
        return "\n".join(lines).strip()
    except Exception:
        # Запасной вариант: парсим sharedStrings из zip-архива.
        import zipfile

        with zipfile.ZipFile(path) as z:
            try:
                xml = z.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
            except KeyError:
                return ""
        strings = re.findall(r"<t[^>]*>(.*?)</t>", xml, flags=re.S)
        return "\n".join(strings).strip()


def read_doc(path: Path) -> str:
    """Чтение старого бинарного .doc: antiword -> libreoffice -> fallback."""
    # Вариант 1: antiword
    try:
        res = subprocess.run(
            ["antiword", "-m", "UTF-8.txt", str(path)],
            capture_output=True, timeout=60,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.decode("utf-8", errors="replace").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Вариант 2: конвертация через libreoffice/soffice в txt
    for exe in ("libreoffice", "soffice"):
        try:
            tmp_dir = BASE_DIR / ".tmp_doc"
            tmp_dir.mkdir(exist_ok=True)
            res = subprocess.run(
                [exe, "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                 "--outdir", str(tmp_dir), str(path)],
                capture_output=True, timeout=120,
            )
            txt_file = tmp_dir / (path.stem + ".txt")
            if res.returncode == 0 and txt_file.exists():
                text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
                txt_file.unlink(missing_ok=True)
                if text:
                    return text
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    # Вариант 3: грубое извлечение печатных последовательностей из бинарника
    data = path.read_bytes()
    words = re.findall(rb"[\x20-\x7e\xd0-\xdf]{4,}", data)
    chunks = []
    for w in words:
        try:
            chunks.append(w.decode("cp1251", errors="ignore"))
        except Exception:
            chunks.append(w.decode("ascii", errors="ignore"))
    return "\n".join(chunks).strip()


def read_xml(path: Path) -> str:
    """Чтение XML как текста с pretty-print, если возможно."""
    raw = read_txt(path)
    try:
        import xml.dom.minidom

        return xml.dom.minidom.parseString(raw).toprettyxml(indent="  ").strip()
    except Exception:
        return raw


def read_json(path: Path) -> str:
    """Чтение JSON как текста с pretty-print, если возможно."""
    raw = read_txt(path)
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except Exception:
        return raw


READERS = {
    ".txt": read_txt,
    ".md": read_txt,
    ".docx": read_docx,
    ".doc": read_doc,
    ".pdf": read_pdf,
    ".xlsx": read_xlsx,
    ".xml": read_xml,
    ".json": read_json,
}


def collect_requirement_files(root: Path) -> list[Path]:
    """Рекурсивный сбор файлов поддерживаемых форматов из root."""
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def extract_content(path: Path) -> str:
    """Извлечение текста из файла с учётом его формата."""
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        return ""
    try:
        text = reader(path) or ""
    except Exception as e:
        return f"[Ошибка чтения файла: {e}]"
    if MAX_FILE_CHARS and len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\n...[файл обрезано по лимиту MAX_FILE_CHARS]..."
    return text


def build_requirements_payload(files: list[Path], root: Path) -> str:
    """Сбор единого документа из содержимого всех файлов требований."""
    blocks = []
    for i, f in enumerate(files, 1):
        rel = f.relative_to(root)
        content = extract_content(f)
        blocks.append(
            f"===== ФАЙЛ {i}: {rel} =====\n{content}\n===== КОНЕЦ ФАЙЛА {i}: {rel} ====="
        )
    return "\n\n".join(blocks)


# ------------------------------ Работа с ИИ ---------------------------------

def ask_ai(prompt_text: str, requirements_text: str) -> str:
    """
    Синхронный запрос к ИИ через OpenAI-совместимый API.

    Функция возвращает управление ТОЛЬКО после того, как модель вернёт
    полный ответ (stream=False, обычный blocking-вызов). Обработка
    результата в вызывающем коде начнётся строго после получения ответа.
    Предусмотрены ретраи при сетевых/серверных ошибках.
    """
    from openai import OpenAI  # импорт здесь, чтобы env читался до создания клиента

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=TIMEOUT)

    user_message = (
        f"{prompt_text}\n\n"
        f"Ниже приведено содержимое файлов с требованиями.\n\n"
        f"{requirements_text}"
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Запрос #{attempt}] Отправка данных в ИИ "
                  f"(модель: {MODEL}, temperature={TEMPERATURE}, max_tokens={MAX_TOKENS})...")
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,  # ждём полный ответ; обработка — только после ответа ИИ
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
            print("[ОТВЕТ ПОЛУЧЕН] ИИ вернул полный ответ, начинаю обработку результата.")
            return answer
        except Exception as e:
            last_error = e
            print(f"[Ошибка] Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                print(f"Повтор через {delay:.0f} с...")
                time.sleep(delay)
    raise RuntimeError(f"Не удалось получить ответ от ИИ после {MAX_RETRIES} попыток: {last_error}")


# ------------------------------ Сохранение ----------------------------------

def save_result(answer: str, files_count: int) -> Path:
    """Сохранение ответа ИИ в файл 'тестирование требований__<дата_время>.ext'."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    out_path = OUTPUT_DIR / f"{RESULT_PREFIX}{timestamp}{OUTPUT_EXTENSION}"

    header = (
        f"# Тестирование требований\n\n"
        f"- Дата и время выполнения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Модель: {MODEL}\n"
        f"- temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}\n"
        f"- Файлов с требованиями обработано: {files_count}\n\n"
        f"---\n\n"
    )
    out_path.write_text(header + answer, encoding="utf-8")
    return out_path


# --------------------------------- main -------------------------------------

def main() -> int:
    print("=" * 70)
    print("Утилита тестирования требований с помощью ИИ")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("ОШИБКА: в .env не задан OPENAI_API_KEY (токен для доступа к ИИ).")
        return 1
    if not INPUT_DIR.exists():
        print(f"ОШИБКА: папка с требованиями не найдена: {INPUT_DIR}")
        return 1
    if not PROMPT_FILE.exists():
        print(f"ОШИБКА: файл промта не найден: {PROMPT_FILE}")
        return 1

    # Шаг 1. Сбор и чтение файлов требований.
    files = collect_requirement_files(INPUT_DIR)
    if not files:
        print(f"В папке {INPUT_DIR} не найдено файлов форматов: "
              f"{', '.join(SUPPORTED_EXTENSIONS)}")
        return 1

    print(f"\n[Шаг 1] Найдено файлов с требованиями: {len(files)}")
    for f in files:
        print(f"   - {f.relative_to(INPUT_DIR)} ({f.suffix})")

    requirements_text = build_requirements_payload(files, INPUT_DIR)
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    print(f"\n[Шаг 1] Содержимое собрано: {len(requirements_text)} символов, "
          f"промт: {len(prompt_text)} символов.")

    # Шаг 2. Запрос к ИИ. Функция блокируется до получения полного ответа.
    print(f"\n[Шаг 2] Отправка данных в ИИ ({OPENAI_BASE_URL})...")
    try:
        answer = ask_ai(prompt_text, requirements_text)
    except Exception as e:
        print(f"\nОШИБКА при запросе к ИИ: {e}")
        return 1

    # Шаг 3. Обработка и сохранение результата — только после получения ответа.
    out_path = save_result(answer, len(files))
    print(f"\n[Шаг 3] Результат ИИ сохранён в файл: {out_path}")
    print("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
