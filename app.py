import gradio as gr
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from transformers import AutoTokenizer
from overlap import unoverlap_list

LABEL_TEXTSPLITTER = "CharacterTextSplitter"
LABEL_RECURSIVE = "RecursiveCharacterTextSplitter"

bert_tokenizer = AutoTokenizer.from_pretrained('prithivMLmods/Betelgeuse-bert-base-uncased')

def length_tokens(txt):
    return len(bert_tokenizer.tokenize(txt))


def extract_separators_from_string(separators_str):
    try:
        separators_str = separators_str.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\") # fix special characters
        separators = separators_str[1:-1].split(", ")
        return [separator.replace('"', "").replace("'", "") for separator in separators]
    except Exception as e:
        raise gr.Error(f"""
        Did not succeed in extracting seperators from string: {separator_str} due to: {str(e)}.
        Please type it in the correct format: "['separator_1', 'separator_2', ...]"
        """)

def change_split_selection(split_selection):
    return (
        gr.Textbox.update(visible=(split_selection==LABEL_RECURSIVE)),
        gr.Radio.update(visible=(split_selection==LABEL_RECURSIVE)),
    )

def chunk(text, length, splitter_selection, separators_str, length_unit_selection, chunk_overlap):
    separators = extract_separators_from_string(separators_str)
    length_function = (length_tokens if "token" in length_unit_selection.lower() else len)
    if splitter_selection == LABEL_TEXTSPLITTER:
        text_splitter = CharacterTextSplitter(
            chunk_size=length,
            chunk_overlap=int(chunk_overlap),
            length_function=length_function,
            strip_whitespace=False,
            is_separator_regex=False,
            separator=" ",
        )
    elif splitter_selection == LABEL_RECURSIVE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=length,
            chunk_overlap=int(chunk_overlap),
            length_function=length_function,
            strip_whitespace=False,
            separators=separators,
        )
    splits = text_splitter.create_documents([text])
    text_splits = [split.page_content for split in splits]
    unoverlapped_text_splits = unoverlap_list(text_splits)
    output = [((split[0], 'Overlap') if split[1] else (split[0], f"Chunk {str(i)}")) for i, split in enumerate(unoverlapped_text_splits)]
    return output

def change_preset_separators(choice):
    text_splitter = RecursiveCharacterTextSplitter()
    if choice == "Default":
        return ["\n\n", "\n", " ", ""]
    elif choice == "Markdown":
        return text_splitter.get_separators_for_language(Language.MARKDOWN)
    elif choice == "Python":
        return text_splitter.get_separators_for_language(Language.PYTHON)
    else:
        raise gr.Error("Choice of preset not recognized.")


EXAMPLE_TEXT = """

Sure! Here is a concise version of the Snake and Ladder game in Python:

```python
import random

class SnakeAndLadder:
    def __init__(self, players):
        self.players = {player: 0 for player in players}
        self.snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
        self.ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}

    def play_game(self):
        while True:
            for player in self.players:
                dice = random.randint(1, 6)
                self.players[player] += dice
                self.players[player] = self.snakes.get(self.players[player], self.players[player])
                self.players[player] = self.ladders.get(self.players[player], self.players[player])
                print(f"{player} rolled {dice} and is now at {self.players[player]}")
                if self.players[player] >= 100:
                    print(f"{player} wins!")
                    return

if __name__ == "__main__":
    game = SnakeAndLadder(["Player 1", "Player 2"])
    game.play_game()
```

### Explanation

1. **Initialization**:
   - `__init__`: Initializes players at position 0 and sets up snakes and ladders.
2. **Playing the Game**:
   - `play_game`: Main game loop that iterates over players.
   - Rolls the dice, updates positions, and checks for snakes and ladders.
   - Prints the player's current position.
   - Checks if a player has won (reached position 100 or more).

This version is shorter while maintaining the core functionality.


"""

    
with gr.Blocks(theme=gr.themes.Soft(text_size='lg', font=["monospace"], primary_hue=gr.themes.colors.green)) as demo:
    text = gr.Textbox(label="Your text 🪶", value=EXAMPLE_TEXT)
    with gr.Row():
        split_selection = gr.Dropdown(
            choices=[
                LABEL_TEXTSPLITTER,
                LABEL_RECURSIVE,
            ],
            value=LABEL_RECURSIVE,
            label="Method to split chunks 🍞",
        )
        separators_selection = gr.Textbox(
            elem_id="textbox_id",
            value=["\n\n", "\n", " ", ""],
            info="Separators used in RecursiveCharacterTextSplitter",
            show_label=False, # or set label to an empty string if you want to keep its space
            visible=True,
        )
        separator_preset_selection = gr.Radio(
            ['Default', 'Python', 'Markdown'],
            label="Choose a preset",
            info="This will apply a specific set of separators to RecursiveCharacterTextSplitter.",
            visible=True,
        )
    with gr.Row():
        length_unit_selection = gr.Dropdown(
            choices=[
                "Character count",
                "Token count (BERT tokens)",
            ],
            value="Character count",
            label="Length function",
            info="How should we measure our chunk lengths?",
        )
        slider_count = gr.Slider(
            50, 500, value=200, step=1, label="Chunk length 📏", info="In the chosen unit."
        )
        chunk_overlap = gr.Slider(
            0, 50, value=10, step=1, label="Overlap between chunks", info="In the chosen unit."
        )
    out = gr.HighlightedText(
        label="Output",
        show_legend=True,
        show_label=False,
        color_map={'Overlap': '#DADADA'}
    )

    split_selection.change(
        fn=change_split_selection,
        inputs=split_selection,
        outputs=[separators_selection, separator_preset_selection],
    )
    separator_preset_selection.change(
        fn=change_preset_separators,
        inputs=separator_preset_selection,
        outputs=separators_selection,
    )
    gr.on(
        [text.change, length_unit_selection.change, separators_selection.change, split_selection.change, slider_count.change, chunk_overlap.change],
        chunk,
        [text, slider_count, split_selection, separators_selection, length_unit_selection, chunk_overlap],
        outputs=out
    )
    demo.load(chunk, inputs=[text, slider_count, split_selection, separators_selection, length_unit_selection, chunk_overlap], outputs=out)
demo.launch()