from dataclasses import dataclass, field


@dataclass
class Turn:
    question: str
    answer: str


class ConversationHistory:
    def __init__(self):
        self.turns: list[Turn] = []

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def format(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n\n".join(lines)
