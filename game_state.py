import copy
import os


class BlockGameState:
    """Core game logic for the block placement game, separated from visualization."""

    def __init__(self):
        # Initialize an empty 8x8 grid
        self.grid = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)]
        self.score = 0
        self.displayed_score = 0  # For visualization smoothing
        self.highest_score = self.get_high_score()

        # Game state tracking
        self.game_over = False
        self.combo_streak = False
        self.combos = [["COMBO 0"], 0]

        # Combo tracking
        self.placements_without_clear = 0  # Count placements without clearing lines
        self.MAX_COMBO_STREAK = (
            3  # Combo resets after this many placements without clears
        )

        # Generate initial shapes
        self.current_shapes = self.generate_valid_shapes()

        # Track changes for reward calculation
        self.last_action_score = 0
        self.last_lines_cleared = 0

        # Define combo naming scheme
        self.combo_names = {
            1: "",
            2: "DOUBLE ",
            3: "TRIPLE ",
            4: "QUAD ",
            5: "PENTA ",
            6: "HEXA ",
        }

    def generate_valid_shapes(self):
        """Generate three valid shapes that can be placed on the current grid."""
        # Use the shape generator from your original code
        from shapes import generate_shapes as generate_valid_shapes

        shapes = generate_valid_shapes(self.grid)

        # Ensure we have a list of 3 shapes
        if shapes is None or not isinstance(shapes, list):
            shapes = [0, 0, 0]

        return shapes

    def is_valid_placement(self, shape_idx, row, col):
        """Check if a shape can be placed at the specified position."""
        # Validate shape index
        if (
            shape_idx < 0
            or shape_idx >= len(self.current_shapes)
            or not self.current_shapes[shape_idx]
            or not hasattr(self.current_shapes[shape_idx], "form")
        ):
            return False

        shape = self.current_shapes[shape_idx]
        size = [len(shape.form), len(shape.form[0])]

        # Check if the shape fits within bounds and doesn't overlap
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    if (
                        row + i >= 8
                        or col + j >= 8
                        or row + i < 0
                        or col + j < 0
                        or self.grid[row + i][col + j]
                    ):
                        return False

        return True

    def get_valid_actions(self):
        """Return all valid (shape_idx, row, col) combinations."""
        valid_actions = []

        for shape_idx in range(len(self.current_shapes)):
            if not self.current_shapes[shape_idx] or not hasattr(
                self.current_shapes[shape_idx], "form"
            ):
                continue

            shape = self.current_shapes[shape_idx]
            size = [len(shape.form), len(shape.form[0])]

            for row in range(8 - size[0] + 1):
                for col in range(8 - size[1] + 1):
                    if self.is_valid_placement(shape_idx, row, col):
                        valid_actions.append((shape_idx, row, col))

        return valid_actions

    def place_shape(self, shape_idx, row, col):
        """Place a shape on the grid and update the game state."""
        if not self.is_valid_placement(shape_idx, row, col):
            return False

        # Get the shape and its dimensions
        shape = self.current_shapes[shape_idx]
        size = [len(shape.form), len(shape.form[0])]

        # Place the shape on the grid
        for i in range(size[0]):
            for j in range(size[1]):
                if shape.form[i][j]:
                    self.grid[row + i][col + j] = shape.color

        # Remove the used shape
        self.current_shapes[shape_idx] = 0

        # Update the grid (clear lines, calculate score)
        lines_cleared = self.update_grid()

        # Update combo streak tracking
        if lines_cleared > 0:
            # Reset the counter when lines are cleared
            self.placements_without_clear = 0
        else:
            # Increment the counter when no lines are cleared
            self.placements_without_clear += 1

            # Reset combo if too many placements without clearing
            if self.placements_without_clear >= self.MAX_COMBO_STREAK:
                self.combo_streak = False
                self.combos[1] = 0
                self.combos[0][-1] = "COMBO 0"
                self.placements_without_clear = 0

        # Generate new shapes if all current shapes are used
        new_shapes_generated = False
        if all(shape == 0 for shape in self.current_shapes):
            self.current_shapes = self.generate_valid_shapes()
            new_shapes_generated = True

        # Check if the game is over
        self.check_game_over()

        return new_shapes_generated

    def update_grid(self):
        """Clear completed rows/columns and update score."""
        self.last_lines_cleared = 0
        score_before = self.score

        # Find rows and columns to delete
        rows_to_delete = []
        cols_to_delete = []

        for i in range(8):
            if all(self.grid[i][j] for j in range(8)):
                rows_to_delete.append(i)

            if all(self.grid[j][i] for j in range(8)):
                cols_to_delete.append(i)

        # Clear the rows
        for row in rows_to_delete:
            for i in range(8):
                self.grid[row][i] = 0

        # Clear the columns
        for col in cols_to_delete:
            for i in range(8):
                self.grid[i][col] = 0

        # Check for all clear bonus
        all_clear = True
        for i in range(8):
            for j in range(8):
                if self.grid[i][j]:
                    all_clear = False
                    break
            if not all_clear:
                break

        # Update score
        lines_cleared = len(rows_to_delete) + len(cols_to_delete)
        self.last_lines_cleared = lines_cleared

        if lines_cleared:
            # Calculate bonus based on combos and number of lines cleared
            bonus = lines_cleared * 10 * (self.combos[1] + 1)
            if lines_cleared > 2:
                bonus *= lines_cleared - 1

            # Add combo information
            combo = self.combo_names.get(lines_cleared, "MULTI ") + f"CLEAR +{bonus}"
            self.combos[0].insert(-1, combo)

            # Add all clear bonus
            if all_clear:
                bonus += 300
                self.combos[0].insert(-1, "ALL CLEAR +300")

            # Limit combo history
            self.combos[0] = self.combos[0][-8:]

            # Update combo count - increase by the number of rows and columns cleared
            self.combos[1] += lines_cleared
            self.combos[0][-1] = f"COMBO {self.combos[1]}"
            self.combo_streak = True

            # Update score
            self.score += bonus

        # Track score change for reward calculation
        self.last_action_score = self.score - score_before

        return lines_cleared

    def check_game_over(self):
        """Check if there are any valid moves left."""
        self.game_over = True

        # Check if any shape can be placed
        for shape_idx in range(len(self.current_shapes)):
            shape = self.current_shapes[shape_idx]
            if not shape or not hasattr(shape, "form"):
                continue

            size = [len(shape.form), len(shape.form[0])]

            for row in range(8 - size[0] + 1):
                for col in range(8 - size[1] + 1):
                    if self.is_valid_placement(shape_idx, row, col):
                        self.game_over = False
                        return

    def get_state(self):
        """Return a dictionary with the current game state."""
        return {
            "grid": copy.deepcopy(self.grid),
            "available_shapes": self.current_shapes,
            "score": self.score,
            "game_over": self.game_over,
            "combo_streak": self.combo_streak,
            "combos": copy.deepcopy(self.combos),
            "placements_without_clear": self.placements_without_clear,
        }

    def get_normalized_state(self):
        """Return a normalized representation of the state for RL."""
        # Create binary grid (1 for filled, 0 for empty)
        binary_grid = [[1 if cell else 0 for cell in row] for row in self.grid]

        # Encode the shapes as binary matrices
        shapes_encoding = []
        for shape in self.current_shapes:
            if shape and hasattr(shape, "form"):
                shapes_encoding.append(shape.form)
            else:
                # Add an empty shape representation
                shapes_encoding.append([[0]])

        return {
            "grid": binary_grid,
            "shapes": shapes_encoding,
            "score": self.score,
            "combo": self.combos[1],
            "placements_without_clear": self.placements_without_clear,
        }

    def get_high_score(self):
        """Get the high score from file."""
        if not os.path.exists("high_score.txt"):
            self.save_score(0)
            return 0

        try:
            with open("high_score.txt", "r") as f:
                score = f.read().strip()
                return int(score) if score.isdigit() else 0
        except (FileNotFoundError, ValueError):
            self.save_score(0)
            return 0

    def save_score(self, score):
        """Save a score to the high score file."""

        with open("high_score.txt", "w") as file:
            file.write(str(score))

    def reset(self):
        """Reset the game state."""

        if self.score > self.highest_score:
            self.save_score(self.score)

        self.grid = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)]
        self.score = 0
        self.displayed_score = 0
        self.game_over = False
        self.combo_streak = False
        self.combos = [["COMBO 0"], 0]
        self.placements_without_clear = 0
        self.current_shapes = self.generate_valid_shapes()
        self.last_action_score = 0
        self.last_lines_cleared = 0
        self.highest_score = self.get_high_score()
