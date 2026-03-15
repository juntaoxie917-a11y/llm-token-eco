from pathlib import Path

DESC = Path("docs/description.md")
FRAG = Path("docs/fragments/description_formula_sections.md")

START = "<!-- formula-sections:start -->"
END = "<!-- formula-sections:end -->"


def main():
    text = DESC.read_text(encoding="utf-8")
    frag = FRAG.read_text(encoding="utf-8").strip()

    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        new_text = f"{head}{START}\n{frag}\n{END}{tail}"
        DESC.write_text(new_text, encoding="utf-8")
        print("[OK] 已替换插槽内容")
        return  # 关键：防止继续追加

    append_block = f"\n\n{START}\n{frag}\n{END}\n"
    DESC.write_text(text.rstrip() + append_block, encoding="utf-8")
    print("[OK] 未找到插槽，已自动追加到文末并创建插槽标记")


if __name__ == "__main__":
    main()