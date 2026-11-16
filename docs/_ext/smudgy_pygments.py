"""Pygments styles for the smudgy docs: one token palette, two modes.

Every token class has the same hue in ``SmudgyLight`` and ``SmudgyDark``;
only the lightness is tuned for the background. No bold, comments italic
in both modes.
"""

from pygments.style import Style
from pygments.token import (
    Comment, Error, Generic, Keyword, Name, Number, Operator, Punctuation,
    String, Text, Whitespace,
)

# semantic role -> (light, dark)
PALETTE = {
    "text": ("#1f2330", "#d9dee8"),
    "muted": ("#5a5f6e", "#9aa1b2"),
    "comment": ("#6e7585", "#8a92a6"),
    "keyword": ("#8839b8", "#c792ea"),
    "string": ("#2f7a3a", "#a3d68a"),
    "number": ("#b0570a", "#f2a65e"),
    "function": ("#2d55c4", "#82a6f5"),
    "builtin": ("#0e7a86", "#62c9d1"),
    "class": ("#946200", "#e8c36a"),
    "error": ("#c62828", "#ff8a80"),
}


def _styles(mode):
    c = {k: v[mode] for k, v in PALETTE.items()}
    return {
        Text: c["text"],
        Whitespace: "",
        Error: c["error"],
        Comment: f"italic {c['comment']}",
        Comment.Preproc: c["keyword"],
        Keyword: c["keyword"],
        Keyword.Constant: c["keyword"],
        Keyword.Namespace: c["keyword"],
        Keyword.Type: c["builtin"],
        Operator: c["muted"],
        Operator.Word: c["keyword"],
        Punctuation: c["muted"],
        Name: c["text"],
        Name.Builtin: c["builtin"],
        Name.Builtin.Pseudo: c["keyword"],
        Name.Function: c["function"],
        Name.Function.Magic: c["function"],
        Name.Class: c["class"],
        Name.Decorator: c["builtin"],
        Name.Exception: c["class"],
        Name.Namespace: c["text"],
        Name.Tag: c["keyword"],
        Name.Attribute: c["function"],
        String: c["string"],
        String.Doc: f"italic {c['comment']}",
        String.Escape: c["number"],
        String.Interpol: c["number"],
        Number: c["number"],
        Generic.Heading: c["function"],
        Generic.Subheading: c["function"],
        Generic.Deleted: c["error"],
        Generic.Inserted: c["string"],
        Generic.Error: c["error"],
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Prompt: c["muted"],
        Generic.Output: c["muted"],
        Generic.Traceback: c["error"],
    }


class SmudgyLight(Style):
    name = "smudgy-light"
    background_color = "#f6f7fa"
    highlight_color = "#3352cc22"
    line_number_color = "#9aa1b2"
    styles = _styles(0)


class SmudgyDark(Style):
    name = "smudgy-dark"
    background_color = "#22222d"
    highlight_color = "#8ba4f033"
    line_number_color = "#5a5f6e"
    styles = _styles(1)
