import io
import base64
import logging
import os
import uuid

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# Font caching for performance.
###############################################################################
def load_fonts(cell_size):
    """
    Attempt to load truetype fonts; if that fails, use the default.
    Returns a tuple: (num_font, letter_font, clue_font)
    """
    try:
        num_font = ImageFont.truetype("arial.ttf", int(cell_size * 0.3))
        letter_font = ImageFont.truetype("arial.ttf", int(cell_size * 0.6))
        clue_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        num_font = ImageFont.load_default()
        letter_font = ImageFont.load_default()
        clue_font = ImageFont.load_default()
    return num_font, letter_font, clue_font

###############################################################################
# Crossword Data Structures and Functions
###############################################################################

class Crossword:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # Grid: cells where words are placed contain a letter; unused cells remain a space ' '
        self.grid = [[' ' for _ in range(cols)] for _ in range(rows)]
        # List of placed words.
        self.placed_words = []
        # Mapping of starting cell coordinates to assigned number.
        self.numbers = {}

    def place_word(self, word, clue, row, col, orientation):
        """Place word in grid if possible; returns True if successful, else False.
           If successful, the grid is modified and the word appended to placed_words.
        """
        word = word.upper()
        if orientation == "across":
            if col < 0 or col + len(word) > self.cols or row < 0 or row >= self.rows:
                return False
            for i, ch in enumerate(word):
                cur = self.grid[row][col + i]
                if cur != ' ' and cur != ch:
                    return False
            for i, ch in enumerate(word):
                self.grid[row][col + i] = ch
            self.placed_words.append({
                "word": word,
                "clue": clue,
                "row": row,
                "col": col,
                "orientation": "across"
            })
            return True

        elif orientation == "down":
            if row < 0 or row + len(word) > self.rows or col < 0 or col >= self.cols:
                return False
            for i, ch in enumerate(word):
                cur = self.grid[row + i][col]
                if cur != ' ' and cur != ch:
                    return False
            for i, ch in enumerate(word):
                self.grid[row + i][col] = ch
            self.placed_words.append({
                "word": word,
                "clue": clue,
                "row": row,
                "col": col,
                "orientation": "down"
            })
            return True

        return False

def is_valid_placement(cw, word, start_row, start_col, orientation):
    """
    Checks if placing `word` at (start_row, start_col) in the given orientation 
    will only form allowed letter sequences. The check enforces that:
    
      - The word fits within the grid.
      - The cell immediately before the word (and after) is empty.
      - For any cell where a letter is to be placed (and isn't already there as an intersection),
        the adjacent perpendicular cells are blank.
    
    This helps prevent forming any extra contiguous words aside from those in the input.
    """
    word = word.upper()
    if orientation == "across":
        # Check horizontal bounds.
        if start_col < 0 or start_col + len(word) > cw.cols or start_row < 0 or start_row >= cw.rows:
            return False
        # Check that the cell immediately to the left of the word is empty (if in-bound)
        if start_col - 1 >= 0 and cw.grid[start_row][start_col - 1] != " ":
            return False
        # Check that the cell immediately after the word is empty (if in-bound)
        if start_col + len(word) < cw.cols and cw.grid[start_row][start_col + len(word)] != " ":
            return False

        for i in range(len(word)):
            current_col = start_col + i
            cell = cw.grid[start_row][current_col]
            # The target must be blank or already matching.
            if cell != " " and cell != word[i]:
                return False

            # If placing a new letter here (cell blank), ensure that cells above and below are blank.
            if cell == " ":
                if start_row - 1 >= 0 and cw.grid[start_row - 1][current_col] != " ":
                    return False
                if start_row + 1 < cw.rows and cw.grid[start_row + 1][current_col] != " ":
                    return False
        return True

    elif orientation == "down":
        # Check vertical bounds.
        if start_row < 0 or start_row + len(word) > cw.rows or start_col < 0 or start_col >= cw.cols:
            return False
        # Check that the cell immediately above the word is empty (if in-bound)
        if start_row - 1 >= 0 and cw.grid[start_row - 1][start_col] != " ":
            return False
        # Check that the cell immediately below the word is empty (if in-bound)
        if start_row + len(word) < cw.rows and cw.grid[start_row + len(word)][start_col] != " ":
            return False

        for i in range(len(word)):
            current_row = start_row + i
            cell = cw.grid[current_row][start_col]
            if cell != " " and cell != word[i]:
                return False

            # If placing a new letter here, ensure that the left and right cells are blank.
            if cell == " ":
                if start_col - 1 >= 0 and cw.grid[current_row][start_col - 1] != " ":
                    return False
                if start_col + 1 < cw.cols and cw.grid[current_row][start_col + 1] != " ":
                    return False
        return True

    return False

def candidate_score(cw, word, row, col, orientation):
    """Score a candidate placement by the number of overlapping letters."""
    score = 0
    word = word.upper()
    if orientation == "across":
        for i, ch in enumerate(word):
            if cw.grid[row][col + i] != ' ':
                score += 1
    elif orientation == "down":
        for i, ch in enumerate(word):
            if cw.grid[row + i][col] != ' ':
                score += 1
    return score

def compute_grid_dimension(word_entries):
    """
    Compute grid dimensions based on word entries.
    Grid dimension is at least 15, or (max word length + half the number of words), whichever is greater.
    """
    if not word_entries:
        return 15
    max_word = max(len(entry["word"]) for entry in word_entries)
    return max(15, max_word + len(word_entries) // 2)

def build_crossword(word_entries):
    """
    Build the crossword puzzle.
    Uses interlocking placements based on common letters between words.
    Ensures that only provided words are formed and all words are connected.
    """
    grid_dim = compute_grid_dimension(word_entries)
    cw = Crossword(grid_dim, grid_dim)
    if not word_entries:
        return cw

    # Sort words descending by length.
    word_entries = sorted(word_entries, key=lambda x: len(x["word"]), reverse=True)

    # Place the longest word at the center horizontally.
    first = word_entries[0]
    word = first["word"]
    clue = first["clue"]
    start_row = grid_dim // 2
    start_col = (grid_dim - len(word)) // 2
    if not cw.place_word(word, clue, start_row, start_col, "across"):
        cw.place_word(word, clue, 0, 0, "across")  # fallback if centering fails

    # For each remaining word, try to interlock with the placed words.
    for entry in word_entries[1:]:
        new_word = entry["word"].upper()
        new_clue = entry["clue"]
        best_candidate = None
        best_score = -1

        # Try to find the best intersecting candidate.
        for placed_entry in cw.placed_words:
            existing_word = placed_entry["word"]
            for i, ch_existing in enumerate(existing_word):
                for j, ch_new in enumerate(new_word):
                    if ch_existing == ch_new:
                        if placed_entry["orientation"] == "across":
                            candidate_row = placed_entry["row"] - j
                            candidate_col = placed_entry["col"] + i
                            candidate_orientation = "down"
                        else:
                            candidate_row = placed_entry["row"] + i
                            candidate_col = placed_entry["col"] - j
                            candidate_orientation = "across"
                        
                        if is_valid_placement(cw, new_word, candidate_row, candidate_col, candidate_orientation):
                            score = candidate_score(cw, new_word, candidate_row, candidate_col, candidate_orientation)
                            if score > best_score:
                                best_score = score
                                best_candidate = (candidate_row, candidate_col, candidate_orientation)

        placed = False
        if best_candidate is not None and best_score > 0:
            row_candidate, col_candidate, orientation_candidate = best_candidate
            # Save grid state and current placement count in case this placement needs to be reverted.
            original_grid = [row[:] for row in cw.grid]
            pre_word_count = len(cw.placed_words)
            placed = cw.place_word(new_word, new_clue, row_candidate, col_candidate, orientation_candidate)
            if placed:
                # Validate that no extra words were formed.
                allowed_words = [entry["word"] for entry in word_entries]
                valid, formed_words = check_extra_words(cw, allowed_words)
                if not valid:
                    cw.grid = original_grid  # Revert grid state.
                    cw.placed_words = cw.placed_words[:pre_word_count]  # Remove invalid placement.
                    placed = False

        # If placement wasn't successful via intersection, try to place it adjacent to existing words.
        if not placed:
            for r in range(cw.rows):
                for c in range(cw.cols - len(new_word) + 1):
                    temp_grid = [row[:] for row in cw.grid]
                    pre_word_count = len(cw.placed_words)
                    if cw.place_word(new_word, new_clue, r, c, "across"):
                        # Ensure the new word is connected to the grid.
                        if is_connected(cw, r, c, len(new_word), "across"):
                            valid, formed_words = check_extra_words(cw, [entry["word"] for entry in word_entries])
                            if valid:
                                placed = True
                                break
                        cw.grid = temp_grid  # revert grid state.
                        cw.placed_words = cw.placed_words[:pre_word_count]  # Remove invalid placement.
                if placed:
                    break

    # Assign numbers to starting cells.
    num = 1
    for r in range(cw.rows):
        for c in range(cw.cols):
            if cw.grid[r][c] != ' ':
                left = cw.grid[r][c - 1] if c - 1 >= 0 else ' '
                up = cw.grid[r - 1][c] if r - 1 >= 0 else ' '
                if left == ' ' or up == ' ':
                    cw.numbers[(r, c)] = num
                    num += 1

    for entry in cw.placed_words:
        pos = (entry["row"], entry["col"])
        entry["number"] = cw.numbers.get(pos, None)
    return cw

def is_connected(cw, row, col, length, orientation):
    """
    Check if the newly placed word is connected to the existing grid.
    A word is considered connected if any of its surrounding cells (perpendicular to its orientation) has a letter.
    """
    if orientation == "across":
        for i in range(length):
            if (row > 0 and cw.grid[row - 1][col + i] != ' ') or (row < cw.rows - 1 and cw.grid[row + 1][col + i] != ' '):
                return True
    elif orientation == "down":
        for i in range(length):
            if (col > 0 and cw.grid[row + i][col - 1] != ' ') or (col < cw.cols - 1 and cw.grid[row + i][col + 1] != ' '):
                return True
    return False

###############################################################################
# Image Drawing Functions
###############################################################################

def create_crossword_image(cw, cell_size=40, show_solution=True):
    """
    Creates a PNG image (base64‑encoded) of the crossword:
      - White for active cells; black for unused cells.
      - Starting cells show a number.
      - Letters are drawn if show_solution is True.
      - Clues are drawn below the grid.
    """
    rows, cols = cw.rows, cw.cols
    grid_width = cols * cell_size
    grid_height = rows * cell_size

    # Extract clues and sort by number.
    across_clues = []
    down_clues = []
    for word_info in cw.placed_words:
        if word_info.get("number") is None:
            continue
        if word_info["orientation"] == "across":
            across_clues.append((word_info["number"], word_info["clue"]))
        else:
            down_clues.append((word_info["number"], word_info["clue"]))
    across_clues.sort(key=lambda x: x[0])
    down_clues.sort(key=lambda x: x[0])

    clues_area = 200
    total_height = grid_height + clues_area

    img = Image.new("RGB", (grid_width, total_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw cells.
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_size, r * cell_size
            if cw.grid[r][c] != ' ':
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill="white")
            else:
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill="black")

    # Draw grid lines.
    for i in range(cols + 1):
        x = i * cell_size
        draw.line([(x, 0), (x, grid_height)], fill="black", width=2)
    for j in range(rows + 1):
        y = j * cell_size
        draw.line([(0, y), (grid_width, y)], fill="black", width=2)

    # Load fonts (cached for performance).
    num_font, letter_font, clue_font = load_fonts(cell_size)

    # Draw numbers and letters.
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            if (r, c) in cw.numbers:
                num_text = str(cw.numbers[(r, c)])
                draw.text((x + 2, y + 2), num_text, fill="black", font=num_font)
            if show_solution and cw.grid[r][c] != ' ':
                letter = cw.grid[r][c]
                bbox = letter_font.getbbox(letter)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                draw.text((x + (cell_size - lw) / 2, y + (cell_size - lh) / 2),
                          letter, fill="black", font=letter_font)

    # Draw clues below the grid.
    clue_margin = 5
    clue_y = grid_height + clue_margin
    draw.text((clue_margin, clue_y), "ACROSS:", fill="black", font=clue_font)
    offset = 20
    for num, clue in across_clues:
        draw.text((clue_margin, clue_y + offset), f"{num}. {clue}", fill="black", font=clue_font)
        offset += 18

    clue_x2 = grid_width // 2 + 10
    draw.text((clue_x2, clue_y), "DOWN:", fill="black", font=clue_font)
    offset2 = 20
    for num, clue in down_clues:
        draw.text((clue_x2, clue_y + offset2), f"{num}. {clue}", fill="black", font=clue_font)
        offset2 += 18

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

###############################################################################
# API Endpoint and Image Generation Functions
###############################################################################

def generate_crossword_images_api(word_clues, cell_size=40):
    """
    Expects input either as a dict with key "words" or as a list of word/clue dicts.
    Returns two base64‑encoded PNG images (unsolved and solved).
    """
    if isinstance(word_clues, dict) and "words" in word_clues:
        words_list = word_clues["words"]
    elif isinstance(word_clues, list):
        words_list = word_clues
    else:
        raise ValueError("Unexpected format for word_clues.")

    cw = build_crossword(words_list)
    img_solution = create_crossword_image(cw, cell_size=cell_size, show_solution=True)
    img_unsolved = create_crossword_image(cw, cell_size=cell_size, show_solution=False)
    return img_unsolved, img_solution

def save_image_from_base64(b64_string, filename):
    """
    Decode the base64 image and write it as a PNG file to the static directory.
    Returns the path to the saved file.
    """
    image_data = base64.b64decode(b64_string)
    # Ensure the static directory exists.
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    file_path = os.path.join(static_dir, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    return file_path

###############################################################################
# Application Factory
###############################################################################

def create_app():
    app = Flask(__name__)

    # Configure logging for production.
    if not app.debug:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)

    @app.route('/generate', methods=['POST'])
    def api_generate():
        try:
            data = request.get_json()
            if not data:
                raise ValueError("No JSON payload provided.")
            unsolved_b64, solved_b64 = generate_crossword_images_api(data, cell_size=40)
            # Generate unique filenames.
            unsolved_filename = f"unsolved_{uuid.uuid4().hex}.png"
            solved_filename = f"solved_{uuid.uuid4().hex}.png"

            # Save images into the static folder.
            save_image_from_base64(unsolved_b64, unsolved_filename)
            save_image_from_base64(solved_b64, solved_filename)

            # Build the URLs to point to the saved images.
            unsolved_url = request.host_url + "static/" + unsolved_filename
            solved_url = request.host_url + "static/" + solved_filename
            return jsonify({
                "unsolved": unsolved_url,
                "solved": solved_url
            })
        except Exception as e:
            app.logger.exception("Error generating crossword:")
            return jsonify({"error": str(e)}), 400

    return app

###############################################################################
# Main Entry Point
###############################################################################
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

def check_extra_words(cw, allowed_words):
    """
    Scans each row and column for contiguous letter sequences.
    Returns (True, found_words) if every found word is in the allowed_words set.
    Otherwise returns (False, found_words) if an extra word is formed.
    """
    allowed_set = set(word.upper() for word in allowed_words)
    found_words = set()

    # Scan rows
    for r in range(cw.rows):
        word = ""
        for c in range(cw.cols):
            if cw.grid[r][c] != " ":
                word += cw.grid[r][c]
            else:
                if len(word) > 1:
                    found_words.add(word)
                word = ""
        if len(word) > 1:
            found_words.add(word)

    # Scan columns
    for c in range(cw.cols):
        word = ""
        for r in range(cw.rows):
            if cw.grid[r][c] != " ":
                word += cw.grid[r][c]
            else:
                if len(word) > 1:
                    found_words.add(word)
                word = ""
        if len(word) > 1:
            found_words.add(word)
    
    # Verify all found words are allowed.
    for word in found_words:
        if word not in allowed_set:
            return False, found_words
    return True, found_words
