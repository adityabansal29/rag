from dataclasses import dataclass


@dataclass
class Turn:
    question: str
    answer: str


class ConversationHistory:
    def __init__(self, max_turns: int = 10):
        self.turns: list[Turn] = []
        self.max_turns = max_turns

    def add(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def format(self) -> str:
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n\n".join(lines)
