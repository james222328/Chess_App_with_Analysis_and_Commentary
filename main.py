##############################################
# Combined Chess App with Enhanced Analysis,
# Commentary, and User Profiles
# + BLEU & ROUGE with Smoothing & More-Aligned Reference
##############################################

import os
import sys
import pygame
import chess
import chess.engine
import threading
import time
import statistics
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import copy
import textwrap
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

# For BLEU & ROUGE
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
# --- NEW IMPORT FOR METEOR ---
from nltk.translate.meteor_score import meteor_score

# Download NLTK dependencies at runtime if needed:
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ============================================
# Part 0: User Profiles and Simple Database
# ============================================

class UserProfile:
    """
    A simple user profile to store user-specific data:
    - username
    - rating
    - past game records (list of tuples, or anything else)
    """
    def __init__(self, username):
        self.username = username
        self.rating = 1200  # for demonstration
        self.game_history = []  # list of (moves, result)

    def add_game(self, moves, result):
        """Stores a finished game in the user's history."""
        self.game_history.append((moves[:], result))

    def update_statistics(self):
        """
        A placeholder to recalc rating or stats based on new games.
        For now, does nothing.
        """
        pass


USER_DATABASE = {}  # In reality, you might use a file or real database

def get_or_create_profile(username: str) -> UserProfile:
    """
    Returns an existing UserProfile if found, otherwise creates a new one.
    """
    if username not in USER_DATABASE:
        USER_DATABASE[username] = UserProfile(username)
    return USER_DATABASE[username]


def user_login_screen(screen, font, prompt_text="Enter Username"):
    """
    Super-minimal 'login' screen that returns a fixed name or 
    could be expanded to accept user keyboard input.
    For demonstration, it returns a static username 
    or can be adapted to an actual input form.
    """
    screen.fill((0, 0, 0))
    txt = font.render(prompt_text, True, (255, 255, 255))
    screen.blit(txt, (50, 50))
    pygame.display.flip()

    # In a real UI, you'd collect user text input here.
    time.sleep(1)  # simulating a pause
    return "Alice" if "White" in prompt_text else "Bob"

# ============================================
# Part 1: Chess Move Analysis and Commentary
# ============================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables and set API key for Google Generative AI
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize the Google Generative AI model using LangChain
# (Lower temperature for more stable responses)
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def launch_engine(engine_path, threads=2, network_file=None):
    try:
        cmd = [engine_path]
        if network_file:
            cmd.append(f"--weights={network_file}")
        logging.info(f"Launching engine with command: {cmd}")
        engine = chess.engine.SimpleEngine.popen_uci(cmd)
        options = {"Threads": threads}
        engine.configure(options)
        logging.info(f"Engine launched: {engine_path}")
        return engine
    except Exception as e:
        logging.error(f"Failed to launch engine at {engine_path}: {e}")
        return None

def get_engine_score(engine, board, time_limit=0.5, depth=None):
    """
    Returns a Score object from the engine's evaluation of the board,
    or None if there's a problem or if the board is already game-over.
    """
    try:
        # If the board is already in a terminal state (checkmate/stalemate),
        # skip engine calls to prevent invalid engine moves.
        if board.is_game_over():
            return None

        info = engine.analyse(board, limit=chess.engine.Limit(time=time_limit, depth=depth))
        return info["score"].pov(board.turn)
    except Exception as e:
        logging.error(f"Error during engine analysis: {e}")
        return None

def stable_evaluation(engine, board, runs=3, time_per_eval=0.5, depth=None):
    """
    Evaluates a board multiple times using the given engine,
    then returns the median of the numeric scores.
    If board is game-over, or no valid scores, returns 0.
    """
    # If board is already in terminal state, skip
    if board.is_game_over():
        return 0

    scores = []
    for i in range(runs):
        s = get_engine_score(engine, board, time_limit=time_per_eval, depth=depth)
        if s is None:
            logging.warning(f"Run {i+1}: Failed to get score.")
            continue
        num = numeric_score(s)
        scores.append(num)
    if not scores:
        logging.warning("No valid scores obtained.")
        return 0
    median_score = statistics.median(scores)
    logging.info(f"Median Score after {runs} runs: {median_score}")
    return median_score

def numeric_score(s):
    if s is None:
        return 0
    try:
        if s.is_mate():
            m = s.mate()
            return 10000 if m > 0 else -10000
        elif hasattr(s.score, 'cp'):
            return s.score.cp
        else:
            return 0
    except Exception as e:
        logging.error(f"Error processing numeric score: {e}")
        return 0

def combine_scores(score1, score2):
    combined_num = (score1 + score2) / 2.0
    if abs(score1) >= 10000:
        return score1
    if abs(score2) >= 10000:
        return score2
    return combined_num

def guess_phase(board):
    pieces_count = len(board.piece_map())
    has_queens = any(p.piece_type == chess.QUEEN for p in board.piece_map().values())
    if pieces_count > 20 and has_queens:
        phase = "Opening"
    elif pieces_count < 10 or not has_queens:
        phase = "Endgame"
    else:
        phase = "Middlegame"
    logging.info(f"Guessed Phase: {phase}")
    return phase

def positional_heuristics(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    
    white_score = 0
    black_score = 0
    for sq, piece in board.piece_map().items():
        val = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_score += val
        else:
            black_score += val
    material_balance = white_score - black_score
    logging.info(f"Material Balance: {material_balance}")

    def is_developed(piece, square):
        rank = chess.square_rank(square)
        if piece.color == chess.WHITE and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            return rank != 0
        elif piece.color == chess.BLACK and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            return rank != 7
        return False

    white_developed = sum(is_developed(p, s) for s, p in board.piece_map().items() if p.color == chess.WHITE)
    black_developed = sum(is_developed(p, s) for s, p in board.piece_map().items() if p.color == chess.BLACK)
    logging.info(f"White Development: {white_developed}, Black Development: {black_developed}")

    def king_safety(color):
        king_square = board.king(color)
        if king_square is None:
            return "N/A"
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        file_pawns = 0
        rank_pawns = 0
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.PAWN:
                if chess.square_file(sq) == king_file:
                    file_pawns += 1
                if chess.square_rank(sq) == king_rank:
                    rank_pawns += 1
        safety_score = 0
        if file_pawns == 0:
            safety_score -= 1
        if rank_pawns == 0:
            safety_score -= 1
        logging.info(f"King Safety for {'White' if color else 'Black'}: {safety_score}")
        return safety_score

    white_king_safety = king_safety(chess.WHITE)
    black_king_safety = king_safety(chess.BLACK)

    return {
        "MaterialBalance": material_balance,
        "WhiteDevelopment": white_developed,
        "BlackDevelopment": black_developed,
        "WhiteKingSafety": white_king_safety,
        "BlackKingSafety": black_king_safety
    }

def interpret_combined_score(score):
    if score >= 10000:
        interpretation = "Checkmate is inevitable."
    elif score <= -10000:
        interpretation = "Checkmate is inevitable against the side to move."
    elif score > 200:
        interpretation = f"Clearly better (+{score/100:.2f})"
    elif score > 50:
        interpretation = f"Slightly better (+{score/100:.2f})"
    elif score > -50:
        interpretation = "Roughly balanced"
    elif score > -200:
        interpretation = f"Slightly worse ({score/100:.2f})"
    else:
        interpretation = f"Clearly worse ({score/100:.2f})"
    logging.info(f"Interpreted Combined Score: {interpretation}")
    return interpretation

def judge_move_quality(score_before, score_after):
    delta = score_after - score_before
    logging.info(f"Delta (Score After - Score Before): {delta}")
    if delta > 100:
        return "Excellent move"
    elif delta > 20:
        return "Good move"
    elif delta > -20:
        return "Reasonable move"
    elif delta > -100:
        return "Dubious move"
    else:
        return "Blunder"

def guess_player_strategy(board_before, move):
    piece_moved = board_before.piece_at(move.from_square)
    if piece_moved is None:
        return "Unclear strategy."
    piece_type = piece_moved.piece_type

    if piece_type == chess.PAWN:
        strategy = "Possibly aiming for space or structural improvements."
    elif piece_type in [chess.KNIGHT, chess.BISHOP]:
        strategy = "Developing or improving minor piece activity."
    elif piece_type == chess.ROOK:
        strategy = "Striving for file control or infiltration."
    elif piece_type == chess.QUEEN:
        strategy = "Centralizing the queen or exerting tactical pressure."
    elif piece_type == chess.KING:
        strategy = "Improving king safety or transitioning to endgame."
    else:
        strategy = "Strategy not immediately clear."
    logging.info(f"Guessed Player Strategy: {strategy}")
    return strategy

def compare_with_alternatives(engine, board, chosen_move, time_limit=0.5, depth=None, multipv=3):
    """
    Returns a short comparison string or "Failed to compare..." if there's an error
    or if board is already game-over.
    """
    if board.is_game_over():
        return "No alternative moves: The game is already over."

    try:
        info = engine.analyse(board, limit=chess.engine.Limit(time=time_limit, depth=depth), multipv=multipv)
        candidate_moves = [(line["pv"][0], numeric_score(line["score"].pov(board.turn))) for line in info]
    except Exception as e:
        logging.error(f"Error during alternative moves comparison: {e}")
        return "Failed to compare with alternative moves."

    logging.info("Top Moves from Engine:")
    for idx, (mv, sc) in enumerate(candidate_moves, start=1):
        logging.info(f"{idx}. Move: {mv}, Score: {sc}")

    chosen_score = None
    for mv, sc in candidate_moves:
        if mv == chosen_move:
            chosen_score = sc
            break

    if chosen_score is None:
        return "The chosen move was not among the engine's top suggestions."

    better_alternatives = [mv for mv, sc in candidate_moves if sc > chosen_score]
    if not better_alternatives:
        return "This move aligns with the engine's top recommendations."
    else:
        return f"There were stronger alternatives, for example {better_alternatives[0]}."

def generate_chess_commentary(fen, analysis, move):
    move_made = analysis['Move']
    move_number = analysis['MoveNumber']
    color_made_move = "White" if (move_number % 2 == 1) else "Black"
    user_data = analysis.get('UserData', '')

    prompt = f"""
{user_data}

You are a professional chess commentator providing concise and insightful commentary on a chess game.

The current position is described by the following FEN: {fen}.

The move just played is {move_made}, which is move number {move_number}.

**We have determined that this move was played by {color_made_move}** 
(because odd move numbers = White, even move numbers = Black).

Additionally, please provide perspective on the next few moves or long-term plans to create a multi-move storyline. If relevant, mention how the upcoming moves or future strategies might unfold.

### **Comprehensive FEN Analysis**:
Thoroughly analyze the provided FEN, ensuring all elements of the position are considered. Your analysis should include, but is not limited to:
- **Piece Placement and Activity**
- **Control of Key Squares**
- **Pawn Structure**
- **King Safety**
- **Material Balance**
- **Potential Tactical Motifs**
- **Strategic Plans**

### **Commentary Structure**:
In your commentary, please address the following points:

1. **Chess Opening or Defense Identification**:
   - **Identification**: Is it a known opening/defense?
   - **Impact Analysis**: Influence on current position.

2. **Position Analysis**:
   - **Material Balance**
   - **Piece Development**
   - **King Safety**
   - **Pawn Structure**
   - **Control of Key Squares**
   - **Advantages/Weaknesses**

3. **Move Evaluation**:
   - **Assessment**: strong/average/mistake?
   - **Rationale**

4. **Strategic/Tactical Considerations**:
   - **Key Plans**
   - **Opportunities/Dangers**

5. **Alternative Moves**:
   - **Comparison**
   - **Impact**

6. **Future Plans (Multi-Move)**:
   - **Short-term Strategies**
   - **Long-term Goals**

### **Style Constraints**:
- Do **not** always begin with "The move {move_made}...". 
- 2–3 sentences total.
- Correctly identify who played {move_made} using odd/even rule.

### **Example Structure**:
- **Overview**: opening/defense
- **Move Impact**: immediate effects
- **Strategic Insight**: future plans

### **Analysis Data**:
- Move number: {move_number}
- Move played: {move_made}
- Material Balance: {analysis.get('MaterialBalance', 'N/A')}
- White Development: {analysis.get('WhiteDevelopment', 'N/A')}, Black Development: {analysis.get('BlackDevelopment', 'N/A')}
- White King Safety: {analysis.get('WhiteKingSafety', 'N/A')}, Black King Safety: {analysis.get('BlackKingSafety', 'N/A')}
- Move Quality: {analysis.get('Quality', 'N/A')}
- Context Before Move: {analysis.get('Contexts', {}).get('BeforeMove', 'N/A')}
- Context After Move: {analysis.get('Contexts', {}).get('AfterMove', 'N/A')}
- Comparison with Alternatives: {analysis.get('Comparison', 'N/A')}
- Player Strategy: {analysis.get('Planning', 'N/A')}
"""

    messages = [
        SystemMessage(content="You are a knowledgeable chess analyzer and commentator."),
        HumanMessage(content=prompt)
    ]

    try:
        response = chat_model.invoke(messages)
        logging.info("Commentary generated successfully.")
        return response.content
    except Exception as e:
        logging.error(f"Error during commentary generation: {e}")
        return "Failed to generate commentary."

def analyze_move_post(board_before, board_after, move, engine_stockfish, engine_lc0,
                      network_file=None, runs=3, time_per_eval=0.5, depth=None, 
                      multipv=3, move_number=None):
    analysis = {}
    try:
        score_before_sf = stable_evaluation(engine_stockfish, board_before, runs=runs,
                                            time_per_eval=time_per_eval, depth=depth)
        score_before_lc0 = stable_evaluation(engine_lc0, board_before, runs=runs,
                                             time_per_eval=time_per_eval, depth=depth)
        score_before = combine_scores(score_before_sf, score_before_lc0)

        score_after_sf = stable_evaluation(engine_stockfish, board_after, runs=runs,
                                           time_per_eval=time_per_eval, depth=depth)
        score_after_lc0 = stable_evaluation(engine_lc0, board_after, runs=runs,
                                            time_per_eval=time_per_eval, depth=depth)
        score_after = combine_scores(score_after_sf, score_after_lc0)

        analysis['score_before'] = score_before
        analysis['score_after'] = score_after
        analysis['Quality'] = judge_move_quality(score_before, score_after)

        # If board is over, interpret combo scores accordingly
        analysis['Contexts'] = {
            "BeforeMove": interpret_combined_score(score_before),
            "AfterMove": interpret_combined_score(score_after)
        }

        analysis['Comparison'] = compare_with_alternatives(
            engine_stockfish, board_before, move,
            time_limit=time_per_eval, depth=depth, multipv=multipv
        )
        analysis['Planning'] = guess_player_strategy(board_before, move)
        heuristics = positional_heuristics(board_before)
        analysis['MaterialBalance'] = heuristics['MaterialBalance']
        analysis['WhiteDevelopment'] = heuristics['WhiteDevelopment']
        analysis['BlackDevelopment'] = heuristics['BlackDevelopment']
        analysis['WhiteKingSafety'] = heuristics['WhiteKingSafety']
        analysis['BlackKingSafety'] = heuristics['BlackKingSafety']
        analysis['Move'] = board_before.san(move)
        analysis['MoveNumber'] = move_number
    except Exception as e:
        logging.error(f"Error during analyze_move_post: {e}")

    return analysis

# ============================================
# Part 2: Pygame Chess App
# ============================================

WIDTH = 800
HEIGHT = 800
ROWS = 8
COLS = 8
SQSIZE = WIDTH // COLS
COMMENTARY_WIDTH = 500
TOTAL_WIDTH = WIDTH + COMMENTARY_WIDTH

class Color:
    def __init__(self, light, dark):
        self.light = light
        self.dark = dark

class Theme:
    def __init__(self, light_bg, dark_bg, 
                 light_trace, dark_trace,
                 light_moves, dark_moves):
        self.bg = Color(light_bg, dark_bg)
        self.trace = Color(light_trace, dark_trace)
        self.moves = Color(light_moves, dark_moves)

class Sound:
    def __init__(self, filepath):
        try:
            self.sound = pygame.mixer.Sound(filepath)
        except Exception as e:
            logging.error(f"Error loading sound {filepath}: {e}")
            self.sound = None

    def play(self):
        if self.sound:
            self.sound.play()

class Dragger:
    def __init__(self):
        self.piece = None
        self.dragging = False
        self.mouseX = 0
        self.mouseY = 0
        self.initial_square = None

    def update_blit(self, surface, piece_texture):
        if self.piece and piece_texture:
            img = piece_texture
            img_center = (self.mouseX, self.mouseY)
            texture_rect = img.get_rect(center=img_center)
            surface.blit(img, texture_rect)

    def update_mouse(self, pos):
        self.mouseX, self.mouseY = pos

    def save_initial(self, square):
        self.initial_square = square

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True

    def undrag_piece(self):
        self.piece = None
        self.dragging = False

class BoardGraphic:
    def __init__(self, board):
        self.board = board
        self.piece_images = {}
        self.load_images()

    def load_images(self):
        pieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        colors = ['white', 'black']
        for color in colors:
            for piece in pieces:
                path = os.path.join(f'assets/images/imgs-80px/{color}_{piece}.png')
                try:
                    image = pygame.image.load(path)
                    self.piece_images[f"{color}_{piece}"] = pygame.transform.scale(image, (SQSIZE, SQSIZE))
                except Exception as e:
                    logging.error(f"Error loading image {path}: {e}")

    def draw(self, surface):
        for square in chess.SQUARES:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            piece = self.board.piece_at(square)
            if piece:
                key = f"{'white' if piece.color else 'black'}_{chess.piece_name(piece.piece_type)}"
                if key in self.piece_images:
                    img = self.piece_images[key]
                    surface.blit(img, pygame.Rect(col*SQSIZE, row*SQSIZE, SQSIZE, SQSIZE))

class Config:
    def __init__(self):
        self.themes = []
        self._add_themes()
        self.idx = 0
        self.theme = self.themes[self.idx]
        self.font = pygame.font.SysFont('Arial', 18, bold=True)
        self.move_sound = Sound(os.path.join('assets/sounds/move.wav'))
        self.capture_sound = Sound(os.path.join('assets/sounds/capture.wav'))

    def change_theme(self):
        self.idx += 1
        self.idx %= len(self.themes)
        self.theme = self.themes[self.idx]

    def _add_themes(self):
        green = Theme((234, 235, 200), (119, 154, 88),
                      (244, 247, 116), (172, 195, 51),
                      (198, 100, 100), (200, 70, 70))
        brown = Theme((235, 209, 166), (165, 117, 80),
                      (245, 234, 100), (209, 185, 59),
                      (198, 100, 100), (200, 70, 70))
        blue = Theme((229, 228, 200), (60, 95, 135),
                     (123, 187, 227), (43, 119, 191),
                     (198, 100, 100), (200, 70, 70))
        gray = Theme((120, 119, 118), (86, 85, 84),
                     (99, 126, 143), (82, 102, 128),
                     (198, 100, 100), (200, 70, 70))

        self.themes = [green, brown, blue, gray]

class Game:
    def __init__(self):
        self.board = chess.Board()
        self.dragger = Dragger()
        self.config = Config()
        self.move_history = []
        self.move_commentary = []
        self.mode = None
        self.engines_initialized = False
        self.engine_stockfish = None
        self.engine_lc0 = None
        self.network_file = "C:/Users/Electrobot/Downloads/Btr C/weights.pb"

        self.lock = threading.Lock()
        self.commentary_queue = queue.Queue()
        self.commentary_thread = threading.Thread(target=self.process_commentary_queue, daemon=True)
        self.commentary_thread.start()

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.system_move_in_progress = False

        self.cvc_running = False
        self.white_profile = None
        self.black_profile = None

    def initialize_engines(self):
        try:
            logging.info("Initializing Stockfish engine...")
            self.engine_stockfish = launch_engine(
                "C:/Users/Electrobot/Downloads/Btr C/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe", 
                threads=2
            )
            if self.engine_stockfish is None:
                raise ValueError("Stockfish engine failed to initialize.")
            logging.info("Stockfish engine initialized successfully.")
            
            if not os.path.exists(self.network_file):
                raise FileNotFoundError(f"Weights file not found at {self.network_file}")

            logging.info("Initializing Lc0 engine...")
            self.engine_lc0 = launch_engine(
                "C:/Users/Electrobot/Downloads/Btr C/lc0.exe", 
                threads=1, 
                network_file=self.network_file
            )
            if self.engine_lc0 is None:
                raise ValueError("Lc0 engine failed to initialize.")
            logging.info("Lc0 engine initialized successfully.")

            # Quick test
            test_board = chess.Board()
            self.engine_stockfish.analyse(test_board, chess.engine.Limit(time=0.1))
            self.engine_lc0.analyse(test_board, chess.engine.Limit(time=0.1))
            
            self.engines_initialized = True
            logging.info("Engines initialized and tested successfully.")
        except Exception as e:
            logging.error(f"Error initializing engines: {e}")
            self.engines_initialized = False

    def shutdown_engines(self):
        if self.engines_initialized:
            try:
                self.engine_stockfish.quit()
                self.engine_lc0.quit()
                logging.info("Engines shut down successfully.")
            except Exception as e:
                logging.error(f"Error shutting down engines: {e}")

    def play_sound(self, captured=False):
        if captured:
            self.config.capture_sound.play()
        else:
            self.config.move_sound.play()

    def analyze_and_commentary(self, move, move_number):
        """
        If the board is already game-over, skip analysis to avoid engine error.
        """
        if self.board.is_game_over():
            return

        board_before = self.board.copy()
        board_before.pop()
        board_after = self.board.copy()
        self.executor.submit(self.perform_analysis_and_commentary, board_before, board_after, move, move_number)

    def perform_analysis_and_commentary(self, board_before, board_after, move, move_number):
        if not self.engines_initialized:
            logging.warning("Engines not initialized. Skipping analysis.")
            return
        if board_after.is_game_over():
            # skip analyzing if game is over
            return
        try:
            with self.lock:
                analysis = analyze_move_post(
                    board_before, board_after, move,
                    self.engine_stockfish, self.engine_lc0,
                    self.network_file, move_number=move_number
                )

            fen = board_after.fen()

            user_data_snippet = ""
            if self.white_profile:
                user_data_snippet += f"White is {self.white_profile.username} (rating {self.white_profile.rating}). "
            if self.black_profile:
                user_data_snippet += f"Black is {self.black_profile.username} (rating {self.black_profile.rating}). "
            analysis['UserData'] = user_data_snippet

            self.commentary_queue.put((fen, analysis, move))
        except Exception as e:
            logging.error(f"Error during analysis and commentary: {e}")

    def generate_reference_commentary(self, fen, move):
        return [
            "White's e4 opens the King's Pawn Game, a very common and flexible opening. This centralizes the pawn, controlling key squares and preparing for further development. The resulting open position offers both sides dynamic possibilities, with White aiming for quick development and central control, while Black will likely respond with a solid defense, perhaps employing a variation of the Sicilian or French Defense depending on their preferences.",
            "Black's e5, a perfectly reasonable response, opens the center and challenges White's control. This isn't a specific named defense yet, but it allows Black to develop naturally while contesting the center. White will likely look to control the center with d4, potentially leading to a more open game with maneuvering and tactical possibilities.",
            "White's development continues with Nf3, a perfectly reasonable move in the early game, likely aiming for a quieter, positional game rather than a sharp, tactical opening. This develops a piece to a central square and prepares for further development, possibly leading to a King's Indian Defense or a similar structure depending on Black's response. White's short-term plan is to complete development, while Black will need to decide whether to challenge White's center or opt for a more passive approach.",
            "Black's Nc6, developing a knight to a central square, is a perfectly reasonable move in an open game, likely aiming for control of the center and potential pressure on c4. This early development doesn't commit Black to any specific opening, leaving options open. White's likely short-term plan involves further development, perhaps with Be3 or Qd2, while Black will look to continue developing pieces and potentially challenge White's center.",
            "White's Bb5, developing the bishop to a strong square, is a perfectly reasonable move in an open game. This early development, likely aiming for control of the center and pressure on the Black queenside, suggests a flexible approach rather than a commitment to a specific opening line. We can expect further piece development and maneuvering in the coming moves, with both sides vying for control of the central squares and potential kingside attacks.",
            "Black's ...Nf6, developing a knight to a central square, is a natural and reasonable move in this open game, likely aiming for a King's Indian Defense or a similar structure. This develops a piece and controls key squares, though White maintains a slight developmental edge. White will likely look to press their advantage by further developing pieces and potentially aiming for a kingside attack, while Black will seek to counter with ...e6 and further development.",
            "White's kingside castling (0-0) is a natural development, bringing the king to safety and connecting rooks in a relatively open game. This move, while sound, doesn't decisively alter the balance; both sides remain poised for further development, with Black likely aiming to challenge White's light-squared bishop and control the center. We can expect a maneuvering phase, with potential for tactical skirmishes around the center and kingside.",
            "Black's ...Nxe4 recaptures the bishop, a reasonable exchange in this open game, likely stemming from a variation of the King's Pawn Opening. This leaves the position relatively balanced, though White now has the initiative to develop further and potentially exploit Black's slightly less coordinated piece placement. White's immediate priorities include developing their queenside and preparing a kingside attack, while Black will aim to consolidate their position and counter White's initiative.",
            "White's Re1, a developing move, suggests a focus on the kingside, potentially aiming for a kingside attack. This follows an open game, with neither side having a clear positional advantage after the early moves. White's next steps will likely involve further development and probing Black's kingside defenses, while Black needs to consolidate and address the potential threat.",
            "White's e4 opens the King's Pawn Game, a very common and flexible opening. This centralizes the pawn, controlling key squares and preparing for further development. Black now faces a multitude of responses, from the Sicilian Defense to more symmetrical setups, and the ensuing battle will likely focus on controlling the center and developing pieces efficiently.",
            "Black's c5, a move often seen in the Sicilian Defense, challenges White's center control. This solid, if unspectacular, move opens up the center and allows Black to fight for space. White will likely respond with d4, aiming to control the center and develop their pieces, leading to a dynamic and potentially open game.",
            "White's d4, pushing the central pawn, continues the King's Pawn Opening, a very common and flexible start. This commits White to a central battle, aiming for space and control of the center. While not the engine's top choice, it's a reasonable move, opening up potential for both tactical and strategic maneuvering, with Black likely responding with ...cxd4 to challenge White's central pawn.",
            "Black's ...Nf6, developing the knight to a central square, is a perfectly reasonable move in this early stage, likely aiming for a solid, possibly hypermodern setup. This develops a piece and controls the center indirectly, though it's not among the engine's top choices, suggesting White might have slight positional advantages if they play precisely. White should focus on developing their own pieces quickly and efficiently, aiming for a strong center and king safety, while looking for opportunities to exploit any weaknesses in Black's pawn structure.",
            "White's h3, a quiet move in an otherwise open game, suggests a focus on controlling the kingside and preparing for kingside castling. While not a top engine choice, it's a reasonable prophylactic measure, preventing ...g5 and potentially opening the h-file later. Black now has several options, including developing their pieces naturally or looking for a pawn break in the center to challenge White's control.",
            "Black's cxd4, reclaiming the center, is a reasonable response, likely part of a more open game. This exchange simplifies the pawn structure, though White maintains a slight initiative due to better piece development. White should aim to control the center and develop their pieces quickly, while Black seeks to consolidate their position and prevent White from gaining a decisive advantage.",
            "White's Bg5, developing the bishop to a strong square, is a reasonable if not optimal choice in an open game. This move, however, falls short of the engine's top recommendations, potentially missing a sharper opportunity to exploit Black's slightly delayed development. Looking ahead, White should focus on further piece development and central control, while Black needs to address the bishop's influence and secure king safety.",
            "Black's ...d6, a move often seen in various defenses, aims to control the center and challenge White's e4 pawn. While not a top engine choice, it's a reasonable attempt to solidify Black's position and prepare for further development. White should focus on developing their pieces efficiently, perhaps targeting the weakened d4 pawn, while Black looks to improve their king safety and exploit any weaknesses in White's pawn structure.",
            "White's c4, pushing their pawn towards the center, is a reasonable attempt to gain space, though not the most precise. This continues an open game, deviating from any established opening, with both sides having developed similarly. Black now has several options, including ...dxc4, aiming for a pawn advantage, or developing their pieces to control the center.",
            "Black's ...dxc3, capturing the pawn, is a reasonable, if not optimal, move in an open game. While it gains a pawn, it opens the c-file for White's pieces and allows White to control the center more effectively. White should look to develop their pieces quickly, aiming for a kingside attack while exploiting the open c-file, while Black needs to consolidate their position and prevent White from gaining a decisive advantage.",
            "White's eleventh move, Qa4+, is a reasonable attempt to exploit a slight imbalance, though not the engine's top choice. This queen move, while risky, immediately creates pressure on Black's kingside and forces a response, potentially opening lines for White's pieces. Black must carefully navigate this tactical storm, while White aims to capitalize on any weaknesses revealed in the ensuing defense.",
            "Black's ...Bd7, developing the bishop to a central square, is a reasonable move in an open game, though it doesn't immediately create any significant threats. The position remains relatively balanced, with both sides needing to focus on further development and king safety. White's active queen and control of the center give them a slight edge, and future plans should involve consolidating this advantage while controlling key squares.",
            "White's queen to c4, a developing move, is a reasonable attempt to centralize and exert pressure, though not the engine's top choice. This position, arising from an irregular opening, sees White slightly behind in development despite material equality. Black should look to counter White's queen pressure while developing their pieces efficiently, perhaps targeting the weak pawn on c3.",
            "Black's 14... Nxe4, reclaiming the center, is a reasonable exchange, though White's earlier development was slightly better. This leaves the position relatively open, with both sides needing to focus on king safety and developing their remaining pieces. White should prioritize castling and coordinating their rooks, while Black aims to control the open files and exploit any weaknesses in White's pawn structure.",
            "White's Nf3, while a reasonable developing move, fails to address Black's strong initiative stemming from an open game, possibly a King's Indian Defense variation. The move allows Black to maintain pressure, and White needs to quickly improve their piece coordination to avoid falling behind. Black's immediate priorities include exploiting the open files and central control, while White must consolidate their position and prepare for a likely counterattack.",
            "Black's ...Nxg5, sacrificing a knight, is a bold attempt to open the position and unleash attacking potential. This unexpected move, deviating from any established opening line, immediately creates tactical complications, with White needing to carefully navigate the ensuing chaos. White's best response likely involves reclaiming the material and then focusing on consolidating their position to counter Black's initiative.",
            "White's Nxg5, a reasonable exchange, continues an open game with neither side having a clear advantage. While White sacrifices a pawn for activity, Black must now carefully navigate the ensuing complications, potentially focusing on king safety and counterplay on the queenside. The long-term plan for White will likely involve consolidating their advantage in the center and exploiting any weaknesses in Black's pawn structure.",
            "What a dramatic finish! This game, seemingly innocuous in its opening, exploded into a tactical maelstrom culminating in a swift and decisive checkmate. Black's Sicilian Defense, a notoriously sharp opening, immediately set the stage for a complex and potentially volatile battle. White's e4 and d4 aimed for a swift center control, a classic approach against the Sicilian, while Black's ...c5 and ...Nf6 demonstrated a commitment to challenging White's central ambitions and developing quickly. The early exchanges, while seemingly even, subtly favored Black's aggressive strategy. Black's ...dxc3, while seemingly a simple recapture, opened up the position and allowed Black to seize the initiative. The middlegame saw a fascinating struggle for control. White's queenside maneuvering, while aiming for long-term positional advantages, proved insufficient to counter Black's sharp tactical play. Black's ...Bg5, a probing move targeting White's kingside, highlighted the inherent risks of White's more passive approach. The key turning point arguably came with ...Nxe4. This seemingly simple knight sacrifice opened up the diagonal for the black queen, allowing for a devastating attack. White's attempt to defend with Nf3 proved too little, too late. Black's subsequent ...Nxg5, while seemingly a simple recapture, further weakened White's kingside defenses. The position was now irrevocably tilted in Black's favor. White's long-term strategic plans, if any existed beyond immediate development, were completely shattered by the force of Black's tactical onslaught. The endgame arrived with breathtaking speed. The final sequence, ...Qa4+ ...Qc4 ...Qxf7#, is a textbook example of a mating combination; Black's queen, unleashed by the earlier sacrifices and tactical maneuvers, delivered a swift and decisive checkmate. White's king, exposed and undefended, had no escape. The game's narrative perfectly illustrates the dangers of neglecting tactical considerations in favor of overly passive positional play. While a slow, positional approach can be effective, it requires a deep understanding of the position and precise execution. In this instance, White's failure to adequately address Black's tactical threats led to a rapid collapse, showcasing the devastating power of a well-executed attack. The final checkmate was not a fluke; it was the inevitable consequence of a series of tactical blunders and a failure to appreciate the dynamic nature of the position.",
            "Black's h6, a seemingly innocuous move, unleashes a devastating attack. While the opening remains unclear, this pawn push opens the h-file, allowing Black's rook to deliver a swift checkmate. White's poor development and king safety, coupled with Black's superior piece coordination, leaves White with no viable defense.",
            "White's queen sacrifice, Qxf7#, delivers a checkmate—a stunning conclusion to what appears to be an open game, possibly deviating from established openings given the rapid exchange of pieces. This decisive move capitalizes on Black's weak kingside defense and poor piece coordination, highlighting a critical blunder by Black. White now enjoys a swift victory, with no further strategic considerations needed.",
            "White continues development with Nc3, aligning with the principles of the Italian Game. This reinforces the center and prepares for potential expansion with d4. Black should now focus on completing development and challenging White’s central control, possibly through ...d6 or ...Bb4.",
            "Black’s ...d6 secures the e5 pawn while opening lines for the dark-squared bishop. This solidifies Black’s position but slightly delays kingside development. White should now aim to advance d4, contesting the center and leveraging their lead in piece activity.",
            "White develops the bishop to c4, entering the Italian Game. This move targets the vulnerable f7 square, putting immediate pressure on Black’s position. Black should consider countering with ...Nf6 to challenge White’s control of the center.",
            "Black develops the bishop to e7, preparing to castle kingside and ensuring king safety. This move also supports the central pawn structure, allowing Black to maintain a solid and resilient position against White’s advancing pieces.",
            "The game transitions into the King's Indian Attack setup, where White develops the knight to f3, aiming for flexibility and control over e5. This move supports White’s king safety while preparing for future central expansion with d4. Black should consider maintaining central presence with ...Nc6 or challenging White’s pawn structure.",
            "Black challenges the center with ...d5 in response to White's pawn thrust in the French Defense. This move equalizes central control and prepares to develop the light-squared bishop. White’s next move will likely define the structure, with exd5 leading to an open game or e5 pushing into a closed, strategic battle.",
            "White develops the knight to c3, a standard move in the Italian Game, supporting the center and opening possibilities for pawn moves like d4. The position remains balanced, but Black must quickly challenge White's lead in development with ...d6 or ...Bb4.",
            "Black captures the pawn on e5, evening out material while also gaining a slight lead in development. The knight on e5 is well-placed for future central activity, but Black should proceed with caution to avoid overextension. White can respond with moves like d4 to reclaim central influence.",
            "An early Bb5 signals a Ruy Lopez setup, with White emphasizing control over the center and attacking the knight on c6. The 5th move by White slightly deviates from the main line but keeps enough tension on d5. Both sides should focus on timely castling and preparing central pawn breaks.",
            "This push in the Caro-Kann Defense reinforces Black’s solid structure against White’s e4 thrust. Black’s 6th move is strong, bolstering the pawn chain while leaving White to solve how to develop the light-squared bishop effectively. White must consider quick piece activity or risk a cramped position.",
            "A French Defense shape emerges with White pushing d4 and e4, but Black’s 7th move slightly weakens the dark squares. The knight maneuver looks average; it gives Black some counterplay but doesn’t challenge White’s space advantage. White will likely keep tension in the center, while Black might seek counterplay via c5.",
            "Sicilian Defense themes are evident after Black’s 8th move, which opens lines for active play on the queenside. Although this move is somewhat committal, it can unleash tactical possibilities on the c-file if White isn’t careful. White should continue developing swiftly and avoid letting Black’s rook become too powerful.",
            "White’s 5th move fits into a Nimzo-Indian Defense plan, pinning the knight on c6 and contesting dark squares. This decision is strong, limiting Black’s standard ...d5 push while preserving White’s bishop pair. Black must find a timely pawn break or risk a passive position in the middlegame.",
            "With this 10th move, Black commits to a Queen’s Gambit Declined structure, consolidating central pawns and maintaining solidity. The move is average—safe but not overly ambitious, giving White slight freedom to expand on the queenside. White should focus on using the semi-open file and supporting a well-timed c5 advance.",
            "White’s 3rd move transitions into a London System setup, aiming for a straightforward development scheme. It’s a solid but somewhat passive approach, potentially letting Black equalize quickly with active piece play. Both sides will look to complete development soon and prepare for middlegame maneuvers.",
            "The 9th move by Black echoes themes of the Slav Defense, fortifying the d5 pawn and restricting White’s central ambitions. This is a strong, practical choice that keeps options open for queenside expansion. White needs to ramp up piece activity or risk being gradually squeezed out of the center.",
            "An English Opening shape emerges, and White’s 2nd move aims to control the center without committing the d- or e-pawns immediately. This move is average: it keeps a flexible setup but hands Black some initiative to challenge the center quickly. Both players should accelerate development to avoid a slow buildup.",
            "Black’s 12th move captures a central pawn in a way reminiscent of the Scotch Game structures, though slightly off the main line. It’s a minor mistake, as it opens the diagonal toward the king and leaves a potential outpost for White’s knight. White should press on the open lines, while Black must handle any immediate threats carefully."
        ]

    def evaluate_commentary_bleu_rouge(self, generated_text, reference_texts):
        """
        Compute BLEU (with smoothing), ROUGE-1 (F1), and METEOR 
        between generated_text and multiple reference_texts.
        """
        # Hypothesis tokens
        gen_tokens = generated_text.split()

        # Reference tokens
        ref_token_lists = [ref.split() for ref in reference_texts]

        # BLEU
        cc = SmoothingFunction()
        bleu_score_val = sentence_bleu(
            ref_token_lists,
            gen_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=cc.method4
        )

        # ROUGE-1-F1
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_scores = []
        for ref in reference_texts:
            rouge_val = scorer.score(ref, generated_text)['rouge1'].fmeasure
            rouge_scores.append(rouge_val)
        rouge_1_f1 = statistics.mean(rouge_scores) if rouge_scores else 0.0

        # METEOR
        meteor_vals = []
        for ref_token_list in ref_token_lists:
            meteor_val = meteor_score([ref_token_list], gen_tokens)
            meteor_vals.append(meteor_val)
        meteor_avg = sum(meteor_vals) / len(meteor_vals) if meteor_vals else 0.0

        return bleu_score_val, rouge_1_f1, meteor_avg

    def process_commentary_queue(self):
        while True:
            try:
                fen, analysis, move = self.commentary_queue.get()
                # If game ended while we were in queue, skip
                if self.board.is_game_over():
                    continue

                commentary = generate_chess_commentary(fen, analysis, move)
                self.move_commentary.append(commentary)

                reference_commentary = self.generate_reference_commentary(fen, move)
                bleu, rouge, meteor = self.evaluate_commentary_bleu_rouge(commentary, reference_commentary)
                logging.info(f"Using multiple references for evaluation.")
                logging.info(f"BLEU={bleu:.4f}, ROUGE-1-F1={rouge:.4f}, METEOR={meteor:.4f}")

                time.sleep(4)
            except Exception as e:
                logging.error(f"Error processing commentary queue: {e}")
            finally:
                self.commentary_queue.task_done()

    def generate_final_summary(self):
        move_sequence = " ".join(self.move_history)

        summary_prompt = f"""
        You are an expert chess commentator. The game has concluded (checkmate or draw).
        Below is the move sequence played:
        {move_sequence}

        Provide a multi-paragraph, high-level summary that references the key turning points,
        long-term strategies that were or were not realized, and overall game narrative.
        Conclude with how the final result emerged. 
        Aim for a cohesive and holistic perspective, tying together early, mid, and final phases.
        """
        messages = [
            SystemMessage(content="You are a knowledgeable chess analyzer and commentator."),
            HumanMessage(content=summary_prompt)
        ]

        try:
            response = chat_model.invoke(messages)
            final_narrative = response.content
        except Exception as e:
            logging.error(f"Error during final summary generation: {e}")
            final_narrative = "Unable to generate final summary."

        self.move_commentary.append(final_narrative)

    def finish_game(self, result="1-0"):
        """
        Called once we detect the game is over. 
        We store the result in user profiles if they exist, 
        then generate a final summary to conclude commentary.
        """
        if self.white_profile:
            self.white_profile.add_game(self.move_history, result)
            self.white_profile.update_statistics()
        if self.black_profile:
            if result == "1-0":
                black_result = "0-1"
            elif result == "0-1":
                black_result = "1-0"
            else:
                black_result = result
            self.black_profile.add_game(self.move_history, black_result)
            self.black_profile.update_statistics()

        logging.info("Game finished. Generating final summary commentary...")
        self.generate_final_summary()

############################################
# Mode selection + main loop
############################################

def mode_selection_screen(screen, font):
    screen.fill((0, 0, 0))
    title = font.render("Select Gameplay Mode", 1, (255, 255, 255))
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))

    modes = ["User vs User", "User vs System", "System vs System"]
    buttons = []
    for idx, mode in enumerate(modes):
        btn_text = font.render(mode, 1, (0, 0, 0))
        btn_rect = pygame.Rect(WIDTH//2 - 100, 200 + idx*100, 200, 50)
        buttons.append((btn_rect, btn_text))

    for btn_rect, btn_text in buttons:
        pygame.draw.rect(screen, (200, 200, 200), btn_rect)
        screen.blit(btn_text, (btn_rect.x + btn_rect.width//2 - btn_text.get_width()//2,
                               btn_rect.y + btn_rect.height//2 - btn_text.get_height()//2))

    pygame.display.flip()
    return buttons, modes

class Main:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
        pygame.display.set_caption('Interactive Chess App with Commentary')
        self.game = Game()
        self.board_graphic = BoardGraphic(self.game.board)
        self.running = True
        self.mode_selected = False
        self.buttons = []
        self.modes = []
        self.selected_mode = None
        self.font = pygame.font.SysFont('Arial', 18, bold=True)

        self.commentary_scroll = 0
        self.move_history_scroll = 0
        self.max_commentary_scroll = 0
        self.max_move_history_scroll = 0

        self.user_profiles_initialized = False

        self.game.initialize_engines()

    def mainloop(self):
        clock = pygame.time.Clock()

        while self.running:
            if not self.mode_selected:
                self.buttons, self.modes = mode_selection_screen(self.screen, self.font)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        self.game.shutdown_engines()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        for idx, (btn_rect, _) in enumerate(self.buttons):
                            if btn_rect.collidepoint(pos):
                                self.selected_mode = self.modes[idx]
                                self.mode_selected = True
                                logging.info(f"Selected mode: {self.selected_mode}")
                                break
                clock.tick(60)
                continue

            if (self.selected_mode in ["User vs User", "User vs System"]) and not self.user_profiles_initialized:
                self.init_user_profiles()
                self.user_profiles_initialized = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.game.shutdown_engines()
                    pygame.quit()
                    sys.exit()

                # Scroll wheel: up = button=4, down = button=5
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # scroll up
                        mouse_pos = pygame.mouse.get_pos()
                        if self.is_mouse_over_commentary(mouse_pos):
                            self.commentary_scroll += 20
                        elif self.is_mouse_over_move_history(mouse_pos):
                            self.move_history_scroll += 20
                    elif event.button == 5:  # scroll down
                        mouse_pos = pygame.mouse.get_pos()
                        if self.is_mouse_over_commentary(mouse_pos):
                            self.commentary_scroll -= 20
                        elif self.is_mouse_over_move_history(mouse_pos):
                            self.move_history_scroll -= 20

                if self.selected_mode == "User vs User":
                    self.handle_user_vs_user(event)
                elif self.selected_mode == "User vs System":
                    self.handle_user_vs_system(event)
                elif self.selected_mode == "System vs System":
                    self.handle_system_vs_system(event)

            self.limit_scroll_offsets()

            self.screen.fill((0, 0, 0))
            self.draw_board()
            self.board_graphic.draw(self.screen)
            self.show_last_move()
            self.show_move_highlights()
            self.show_hover()
            self.show_commentary()
            self.show_move_history()

            # In "System vs System," show both Start/Next/Previous buttons
            # Otherwise, show "Previous Move" for user-based modes
            if self.selected_mode == "System vs System":
                self.draw_system_vs_system_buttons()
            else:
                self.draw_previous_move_button_user_modes()

            if self.game.dragger.dragging:
                piece_texture = self.get_piece_texture()
                self.game.dragger.update_blit(self.screen, piece_texture)

            pygame.display.update()
            clock.tick(60)

    def init_user_profiles(self):
        if self.selected_mode == "User vs User":
            white_username = user_login_screen(self.screen, self.font, prompt_text="Enter White's Username")
            black_username = user_login_screen(self.screen, self.font, prompt_text="Enter Black's Username")
            self.game.white_profile = get_or_create_profile(white_username)
            self.game.black_profile = get_or_create_profile(black_username)
        elif self.selected_mode == "User vs System":
            white_username = user_login_screen(self.screen, self.font, prompt_text="Enter your Username (White)")
            self.game.white_profile = get_or_create_profile(white_username)
            self.game.black_profile = None

        self.user_profiles_initialized = True

    def draw_system_vs_system_buttons(self):
        if not self.game.cvc_running:
            label = "Start Game"
            btn_text = self.font.render(label, True, (0, 0, 0))
            btn_rect = pygame.Rect(WIDTH // 2 - 50, 10, 100, 30)
            pygame.draw.rect(self.screen, (200, 200, 200), btn_rect)
            self.screen.blit(btn_text, (
                btn_rect.x + (btn_rect.width - btn_text.get_width()) // 2,
                btn_rect.y + (btn_text.get_height() // 2)
            ))
        else:
            prev_label = "Previous Move"
            prev_text = self.font.render(prev_label, True, (0, 0, 0))
            prev_rect = pygame.Rect(WIDTH // 2 - 105, 10, 100, 30)
            pygame.draw.rect(self.screen, (200, 200, 200), prev_rect)
            self.screen.blit(prev_text, (
                prev_rect.x + (prev_rect.width - prev_text.get_width()) // 2,
                prev_rect.y + (prev_rect.height - prev_text.get_height()) // 2
            ))

            next_label = "Next Move"
            next_text = self.font.render(next_label, True, (0, 0, 0))
            next_rect = pygame.Rect(WIDTH // 2 + 5, 10, 100, 30)
            pygame.draw.rect(self.screen, (200, 200, 200), next_rect)
            self.screen.blit(next_text, (
                next_rect.x + (next_rect.width - next_text.get_width()) // 2,
                next_rect.y + (next_rect.height - next_text.get_height()) // 2
            ))

    def draw_previous_move_button_user_modes(self):
        # We'll place it similarly to system vs system's "Previous Move" button, at top center.
        prev_label = "Previous Move"
        prev_text = self.font.render(prev_label, True, (0, 0, 0))
        prev_rect = pygame.Rect(WIDTH // 2 - 50, 10, 100, 30)
        pygame.draw.rect(self.screen, (200, 200, 200), prev_rect)
        self.screen.blit(prev_text, (
            prev_rect.x + (prev_rect.width - prev_text.get_width()) // 2,
            prev_rect.y + (prev_rect.height - prev_text.get_height()) // 2
        ))

    def handle_system_vs_system(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()

            if not self.game.cvc_running:
                btn_rect = pygame.Rect(WIDTH // 2 - 50, 10, 100, 30)
                if btn_rect.collidepoint(mouse_pos):
                    self.game.cvc_running = True
            else:
                prev_rect = pygame.Rect(WIDTH // 2 - 105, 10, 100, 30)
                next_rect = pygame.Rect(WIDTH // 2 + 5, 10, 100, 30)

                if prev_rect.collidepoint(mouse_pos):
                    self.system_move_previous()
                elif next_rect.collidepoint(mouse_pos):
                    if (self.game.engines_initialized 
                        and not self.game.board.is_game_over()
                        and not self.game.system_move_in_progress):
                        self.system_move()

    def handle_user_vs_user(self, event):
        # Check if "Previous Move" clicked
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            prev_rect = pygame.Rect(WIDTH // 2 - 50, 10, 100, 30)
            if prev_rect.collidepoint(mouse_pos):
                self.system_move_previous()
        self.handle_event(event)

    def handle_user_vs_system(self, event):
        # Check if "Previous Move" clicked
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            prev_rect = pygame.Rect(WIDTH // 2 - 50, 10, 100, 30)
            if prev_rect.collidepoint(mouse_pos):
                self.system_move_previous()

        self.handle_event(event)
        # After user move, if game ended do finish_game
        # else if still going and black to move, system moves
        if self.selected_mode == "User vs System" and self.game.engines_initialized:
            if self.game.board.is_game_over():
                result = self.game.board.result()
                if result not in ["1-0", "0-1", "1/2-1/2"]:
                    result = "1/2-1/2"
                self.game.finish_game(result)
            elif self.game.board.turn == chess.BLACK and not self.game.system_move_in_progress:
                self.system_move()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.game.dragger.update_mouse(event.pos)
            clicked_square = self.get_square_from_pos(event.pos)
            if clicked_square is not None:
                piece = self.game.board.piece_at(clicked_square)
                if piece and piece.color == self.game.board.turn:
                    self.game.dragger.save_initial(clicked_square)
                    self.game.dragger.drag_piece(piece)

        elif event.type == pygame.MOUSEMOTION:
            self.game.dragger.update_mouse(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.game.dragger.dragging:
                released_square = self.get_square_from_pos(event.pos)
                initial_square = self.game.dragger.initial_square
                if released_square is not None and initial_square is not None:
                    move = chess.Move(initial_square, released_square)
                    if move in self.game.board.legal_moves:
                        move_san = self.game.board.san(move)
                        captured = self.game.board.is_capture(move)
                        self.game.board.push(move)
                        self.game.move_history.append(move_san)
                        self.game.play_sound(captured)
                        move_number = len(self.game.move_history)
                        self.game.analyze_and_commentary(move, move_number)

                        # If game ended immediately after user's move:
                        if self.game.board.is_game_over():
                            final_result = self.game.board.result()
                            if final_result not in ["1-0", "0-1", "1/2-1/2"]:
                                final_result = "1/2-1/2"
                            self.game.finish_game(final_result)

            self.game.dragger.undrag_piece()

    def system_move(self):
        if self.game.board.is_game_over():
            # Do not attempt system move if game is done
            return

        def task():
            if not self.game.engines_initialized:
                logging.warning("Engines not initialized. Cannot perform system move.")
                return
            try:
                if self.selected_mode not in ["User vs System", "System vs System"]:
                    logging.warning(f"System move attempted in mode: {self.selected_mode}")
                    return
                if self.game.board.is_game_over():
                    return

                with self.game.lock:
                    result = self.game.engine_stockfish.play(self.game.board, chess.engine.Limit(time=0.5))
                move = result.move
                if move is not None:
                    move_san = self.game.board.san(move)
                    captured = self.game.board.is_capture(move)
                    self.game.board.push(move)
                    self.game.move_history.append(move_san)
                    self.game.play_sound(captured)
                    move_number = len(self.game.move_history)
                    self.game.analyze_and_commentary(move, move_number)

                    # If game ends after system move
                    if self.game.board.is_game_over():
                        final_result = self.game.board.result()
                        if final_result not in ["1-0", "0-1", "1/2-1/2"]:
                            final_result = "1/2-1/2"
                        self.game.finish_game(final_result)

            except Exception as e:
                logging.error(f"Error during system move: {e}")
            finally:
                self.game.system_move_in_progress = False

        self.game.system_move_in_progress = True
        threading.Thread(target=task, daemon=True).start()

    def system_move_previous(self):
        if len(self.game.board.move_stack) > 0:
            self.game.board.pop()
            if self.game.move_history:
                self.game.move_history.pop()
            if self.game.move_commentary:
                self.game.move_commentary.pop()

    def get_square_from_pos(self, pos):
        x, y = pos
        if x >= WIDTH or y >= HEIGHT:
            return None
        col = x // SQSIZE
        row = 7 - (y // SQSIZE)
        return chess.square(col, row)

    def get_piece_texture(self):
        if self.game.dragger.piece:
            piece = self.game.dragger.piece
            key = f"{'white' if piece.color else 'black'}_{chess.piece_name(piece.piece_type)}"
            return self.board_graphic.piece_images.get(key, None)
        return None

    def show_last_move(self):
        if self.game.board.move_stack:
            last_move = self.game.board.peek()
            start_square = last_move.from_square
            end_square = last_move.to_square
            start_pos = (chess.square_file(start_square)*SQSIZE, 
                         (7 - chess.square_rank(start_square))*SQSIZE)
            end_pos = (chess.square_file(end_square)*SQSIZE, 
                       (7 - chess.square_rank(end_square))*SQSIZE)
            pygame.draw.rect(self.screen, (255, 255, 0),
                             pygame.Rect(*start_pos, SQSIZE, SQSIZE), 3)
            pygame.draw.rect(self.screen, (255, 255, 0),
                             pygame.Rect(*end_pos, SQSIZE, SQSIZE), 3)

    def show_move_highlights(self):
        if self.game.board.move_stack:
            for move in list(self.game.board.move_stack)[-5:]:
                start_square = move.from_square
                end_square = move.to_square
                start_pos = (chess.square_file(start_square)*SQSIZE,
                             (7 - chess.square_rank(start_square))*SQSIZE)
                end_pos = (chess.square_file(end_square)*SQSIZE,
                           (7 - chess.square_rank(end_square))*SQSIZE)
                pygame.draw.rect(self.screen, (173, 216, 230),
                                 pygame.Rect(*start_pos, SQSIZE, SQSIZE), 3)
                pygame.draw.rect(self.screen, (173, 216, 230),
                                 pygame.Rect(*end_pos, SQSIZE, SQSIZE), 3)

    def show_hover(self):
        pos = (self.game.dragger.mouseX, self.game.dragger.mouseY)
        square = self.get_square_from_pos(pos)
        if square is not None:
            row = chess.square_rank(square)
            col = chess.square_file(square)
            pygame.draw.rect(self.screen, (180, 180, 180),
                             pygame.Rect(col*SQSIZE, (7 - row)*SQSIZE, SQSIZE, SQSIZE),
                             3)

    def show_commentary(self):
        commentary_rect = pygame.Rect(WIDTH, 0, COMMENTARY_WIDTH, HEIGHT // 2)
        pygame.draw.rect(self.screen, (50, 50, 50), commentary_rect)
        self.screen.set_clip(commentary_rect)
        y_offset = 10 - self.commentary_scroll
        wrap_width = 60

        for idx, commentary in enumerate(self.game.move_commentary):
            move_num = idx + 1
            wrapped_text = textwrap.wrap(commentary, width=wrap_width)

            if not wrapped_text:
                wrapped_text = [" "]

            for line_idx, line in enumerate(wrapped_text):
                if line_idx == 0:
                    text = f"{move_num}. {line}"
                    lbl = self.game.config.font.render(text, True, (255, 255, 255))
                    self.screen.blit(lbl, (WIDTH + 10, y_offset))
                else:
                    indent = "   "
                    text = f"{indent}{line}"
                    lbl = self.game.config.font.render(text, True, (255, 255, 255))
                    self.screen.blit(lbl, (WIDTH + 40, y_offset))
                y_offset += 25

        self.screen.set_clip(None)
        total_lines = 0
        for commentary in self.game.move_commentary:
            wrapped = textwrap.wrap(commentary, width=wrap_width)
            total_lines += len(wrapped)
        
        self.max_commentary_scroll = max(0, (total_lines * 25) - (HEIGHT // 2 - 20))
        if self.commentary_scroll < 0:
            self.commentary_scroll = 0
        elif self.commentary_scroll > self.max_commentary_scroll:
            self.commentary_scroll = self.max_commentary_scroll

    def show_move_history(self):
        move_history_rect = pygame.Rect(WIDTH, HEIGHT // 2, COMMENTARY_WIDTH, HEIGHT // 2)
        pygame.draw.rect(self.screen, (30, 30, 30), move_history_rect)
        self.screen.set_clip(move_history_rect)
        y_offset = HEIGHT // 2 + 10 - self.move_history_scroll
        wrap_width = 20

        for idx, move_str in enumerate(self.game.move_history):
            move_num = idx + 1
            wrapped_text = textwrap.wrap(move_str, width=wrap_width)

            if not wrapped_text:
                wrapped_text = [" "]

            for line_idx, line in enumerate(wrapped_text):
                if line_idx == 0:
                    text = f"{move_num}. {line}"
                    lbl = self.game.config.font.render(text, True, (255, 255, 255))
                    self.screen.blit(lbl, (WIDTH + 10, y_offset))
                else:
                    indent = "   "
                    text = f"{indent}{line}"
                    lbl = self.game.config.font.render(text, True, (255, 255, 255))
                    self.screen.blit(lbl, (WIDTH + 40, y_offset))
                y_offset += 25

        self.screen.set_clip(None)
        total_lines = 0
        for move_str in self.game.move_history:
            wrapped = textwrap.wrap(move_str, width=wrap_width)
            total_lines += len(wrapped)
        
        self.max_move_history_scroll = max(0, (total_lines * 25) - (HEIGHT // 2 - 20))
        if self.move_history_scroll < 0:
            self.move_history_scroll = 0
        elif self.move_history_scroll > self.max_move_history_scroll:
            self.move_history_scroll = self.max_move_history_scroll

    def is_mouse_over_commentary(self, pos):
        x, y = pos
        return WIDTH < x < TOTAL_WIDTH and 0 <= y < HEIGHT // 2

    def is_mouse_over_move_history(self, pos):
        x, y = pos
        return WIDTH < x < TOTAL_WIDTH and HEIGHT // 2 <= y < HEIGHT

    def wrap_text(self, text, font, max_width):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def limit_scroll_offsets(self):
        if self.commentary_scroll < 0:
            self.commentary_scroll = 0
        elif self.commentary_scroll > self.max_commentary_scroll:
            self.commentary_scroll = self.max_commentary_scroll

        if self.move_history_scroll < 0:
            self.move_history_scroll = 0
        elif self.move_history_scroll > self.max_move_history_scroll:
            self.move_history_scroll = self.max_move_history_scroll

    def draw_board(self):
        theme = self.game.config.theme
        for row in range(ROWS):
            for col in range(COLS):
                color = theme.bg.light if (row + col) % 2 == 0 else theme.bg.dark
                pygame.draw.rect(self.screen, color, pygame.Rect(col*SQSIZE, row*SQSIZE, SQSIZE, SQSIZE))

                if col == 0:
                    lbl = self.game.config.font.render(str(ROWS - row), True, (0, 0, 0))
                    self.screen.blit(lbl, (5, row * SQSIZE + 5))

                if row == 7:
                    lbl = self.game.config.font.render(chess.square_name(chess.square(col, row))[0], True, (0, 0, 0))
                    self.screen.blit(lbl, (col * SQSIZE + SQSIZE - 20, HEIGHT - 20))

if __name__ == "__main__":
    main = Main()
    main.mainloop()
