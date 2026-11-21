
import re


def clean_text(text):
    """Clean extracted PDF text while preserving question structure."""

    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        # Remove decorative PDF characters
        line = line.strip("*").strip()

        if line:
            lines.append(line)

    return lines


def remove_pdf_artifacts(text):
    """Remove unwanted PDF headers, footers and registration details."""

    patterns = [
        r"\s*APJ ABDUL KALAM TECHNOLOGICAL UNIVERSITY.*$",
        r"\s*Reg(?:istration)?\s*No\.?\s*:.*$",
        r"\s*Name\s*:.*$",
        r"\s*Pages?\s*:\s*\d+.*$",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    # Remove stray decorative characters
    text = re.sub(r"\*+", "", text)

    return text.strip()


def extract_questions(text):
    """Extract Part A and Part B questions from question paper text."""

    lines = clean_text(text)

    questions = []

    current_part = None
    current_module = None
    current_question = None
    current_alternative = 1
    last_question_number = None
    pending_number = None

    def save_question():
        nonlocal current_question

        if current_question:
            cleaned_question = remove_pdf_artifacts(
                current_question["text"]
            )

            current_question["text"] = cleaned_question

            if cleaned_question:
                questions.append(current_question)

        current_question = None

    def start_question(number, subpart, content):
        nonlocal current_question, last_question_number

        save_question()

        last_question_number = number

        current_question = {
            "part": current_part,
            "module": current_module,
            "question_number": number,
            "subpart": subpart,
            "alternative": current_alternative,
            "text": content.strip()
        }

    for line in lines:

        upper_line = line.upper().strip()

        # -----------------------------------------
        # Detect Part A
        # -----------------------------------------
        if re.search(r"\bPART\s*[-:]?\s*A\b", upper_line):
            save_question()

            current_part = "A"
            current_module = None
            current_alternative = 1
            pending_number = None

            continue

        # -----------------------------------------
        # Detect Part B
        # -----------------------------------------
        if re.search(r"\bPART\s*[-:]?\s*B\b", upper_line):
            save_question()

            current_part = "B"
            current_module = None
            current_alternative = 1
            pending_number = None

            continue

        # -----------------------------------------
        # Detect Modules I, II, III, IV, V
        # -----------------------------------------
        module_match = re.search(
            r"\bMODULE\s+(I|II|III|IV|V)\b",
            upper_line
        )

        if module_match:
            save_question()

            current_module = module_match.group(1)
            current_alternative = 1
            pending_number = None

            continue

        # -----------------------------------------
        # Detect alternative questions (OR)
        # -----------------------------------------
        if upper_line in ("OR", "OR:"):
            save_question()

            current_alternative += 1
            pending_number = None

            continue

        # -----------------------------------------
        # Ignore common instructions and page numbers
        # -----------------------------------------
        if upper_line.startswith("ANSWER "):
            continue

        if re.match(r"^PAGE\s+\d+", upper_line):
            continue

        # -----------------------------------------
        # Part A: standalone question number
        # Example: 1.
        # -----------------------------------------
        standalone_number = re.match(
            r"^(\d{1,2})[.)]?$",
            line
        )

        if standalone_number and current_part == "A":
            save_question()

            pending_number = int(
                standalone_number.group(1)
            )

            continue

        # -----------------------------------------
        # Part A: numbered questions
        # Examples:
        # 1. Explain cloud computing
        # 1) Explain cloud computing
        # 1 Explain cloud computing
        # -----------------------------------------
        if current_part == "A":

            part_a_match = re.match(
                r"^(\d{1,2})[.)]\s*(.+)$",
                line
            )

            if not part_a_match:
                part_a_match = re.match(
                    r"^(\d{1,2})\s+(.+)$",
                    line
                )

            if part_a_match:
                number = int(part_a_match.group(1))
                content = part_a_match.group(2)

                start_question(
                    number,
                    None,
                    content
                )

                pending_number = None
                continue

            # Question number and text on separate lines
            if pending_number is not None:

                start_question(
                    pending_number,
                    None,
                    line
                )

                pending_number = None
                continue

        # -----------------------------------------
        # Part B: numbered subquestions
        # Example: 11 a) Explain NIST architecture
        # -----------------------------------------
        part_b_match = re.match(
            r"^(\d{1,2})\s+([a-zA-Z])\)\s*(.+)$",
            line
        )

        if part_b_match and current_part == "B":

            number = int(part_b_match.group(1))
            subpart = part_b_match.group(2).lower()
            content = part_b_match.group(3)

            start_question(
                number,
                subpart,
                content
            )

            continue

        # -----------------------------------------
        # Part B: subpart on separate line
        # Example: b) Explain cloud security
        # -----------------------------------------
        subpart_match = re.match(
            r"^([a-zA-Z])\)\s*(.+)$",
            line
        )

        if subpart_match and current_part == "B":

            save_question()

            current_question = {
                "part": current_part,
                "module": current_module,
                "question_number": last_question_number,
                "subpart": subpart_match.group(1).lower(),
                "alternative": current_alternative,
                "text": subpart_match.group(2).strip()
            }

            continue

        # -----------------------------------------
        # Add wrapped lines to current question
        # -----------------------------------------
        if current_question:
            current_question["text"] += " " + line

    # Save final question
    save_question()

    return questions
