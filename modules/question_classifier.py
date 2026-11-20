
import re


def clean_text(text):
    """Clean extracted PDF text while preserving question structure."""

    text = text.replace("\xa0", " ")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return lines


def extract_questions(text):

    lines = clean_text(text)

    questions = []

    current_part = None
    current_module = None
    current_question = None
    current_subpart = None
    alternative = 1

    def save_question():
        nonlocal current_question

        if current_question:
            current_question["text"] = (
                current_question["text"].strip()
            )

            if current_question["text"]:
                questions.append(current_question)

            current_question = None

    def start_question(number, subpart, content):

        nonlocal current_question, current_subpart

        save_question()

        current_subpart = subpart

        current_question = {
            "part": current_part,
            "module": current_module,
            "question_number": number,
            "subpart": subpart,
            "alternative": alternative,
            "text": content.strip()
        }

    for line in lines:

        upper_line = line.upper()

        # Detect Part A / Part B
        if re.search(r"\bPART\s+A\b", upper_line):
            save_question()
            current_part = "A"
            current_module = None
            continue

        if re.search(r"\bPART\s+B\b", upper_line):
            save_question()
            current_part = "B"
            current_module = None
            continue

        # Detect modules
        module_match = re.search(
            r"\bMODULE\s+(I|II|III|IV|V)\b",
            upper_line
        )

        if module_match:
            save_question()
            current_module = module_match.group(1)
            alternative = 1
            continue

        # Detect OR (alternative questions)
        if upper_line == "OR":
            save_question()
            alternative += 1
            continue

        # Ignore instructions and page numbers
        if upper_line.startswith("ANSWER "):
            continue

        if re.match(r"^PAGE\s+\d+", upper_line):
            continue

        # Detect numbered questions: 11 a), 1, 12 a), etc.
        question_match = re.match(
            r"^(\d{1,2})\s*(?:([a-zA-Z])\))?\s+(.+)$",
            line
        )

        if question_match:

            number = int(question_match.group(1))
            subpart = question_match.group(2)
            content = question_match.group(3)

            start_question(number, subpart, content)
            continue

        # Detect subquestions: b), c), etc.
        subpart_match = re.match(
            r"^([a-zA-Z])\)\s*(.+)$",
            line
        )

        if subpart_match and current_part == "B":

            save_question()

            current_subpart = subpart_match.group(1)

            current_question = {
                "part": current_part,
                "module": current_module,
                "question_number": (
                    questions[-1]["question_number"]
                    if questions else None
                ),
                "subpart": current_subpart,
                "alternative": alternative,
                "text": subpart_match.group(2).strip()
            }

            continue

        # Add wrapped lines to current question
        if current_question:
            current_question["text"] += " " + line

    save_question()

    return questions
