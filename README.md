# Chess App with Enhanced Analysis, Commentary, and User Profiles

An interactive **Pygame**-based Chess application featuring **multi-engine analysis** (Stockfish & Lc0), **AI-generated commentary** (via Google Generative AI), **BLEU/ROUGE/METEOR** scoring for commentary evaluation, and **user profiles** for tracking games and ratings.

## Features

1. **Interactive Chess Board**: Play locally (User vs User) or challenge the system (User vs System). There's also a System vs System mode for engine-to-engine battles.  
2. **Multi-Engine Analysis**: Integrates Stockfish and Lc0 to provide combined analysis scores and move suggestions.  
3. **AI-Powered Commentary**: Uses Google Generative AI to generate real-time, context-aware commentary after each move.  
4. **Automated Evaluation**: BLEU, ROUGE, and METEOR scores evaluate the generated commentary against reference texts.  
5. **User Profiles**: Basic profile tracking (username, rating, game history).  
6. **Multi-Threaded**: Employs Python’s `threading` and `ThreadPoolExecutor` for efficient background engine analysis and commentary generation.  
7. **In-Game Visuals**: Highlights last moves, shows move history, and logs commentary with scrolling panels in the GUI.

## Requirements

- **Python 3.8+**
- [Pygame](https://www.pygame.org/) for the GUI
- [python-chess](https://pypi.org/project/python-chess/) for board logic and engine integrations
- [nltk](https://www.nltk.org/) (Natural Language Toolkit) for tokenization and BLEU/METEOR scoring
- [rouge-score](https://pypi.org/project/rouge-score/) for ROUGE metrics
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) & [google-generativeai](https://pypi.org/project/google-generativeai/) for commentary generation
- Any additional libraries used in your code (e.g., `dataclasses`, `dotenv`, etc.)

**Stockfish and Lc0**  
- Download or place the Stockfish and Lc0 executables where your code references them.
- Update the paths in the code (`engine_stockfish` and `engine_lc0`) to match your local setup.

## Setup

1. **Configure Google Generative AI**:
   - Sign up for Google PaLM/bard API access.
   - Obtain your API key and store it in an `.env` file or as an environment variable:
     ```
     GOOGLE_API_KEY=YOUR_KEY_HERE
     ```
   - Make sure your code can read this key (e.g., using `load_dotenv()`).

2. **Check engine paths**:  
   - In your code, verify the `engine_stockfish` and `engine_lc0` paths match where you installed your chess engines.



- A **GUI** should open, letting you choose a gameplay mode:
  - **User vs User**: Two human players on the same system.
  - **User vs System**: One human player vs the Stockfish engine.
  - **System vs System**: Stockfish vs Lc0 in a fully automated match.

- During gameplay, commentary is generated after each move, and the move history + commentary can be scrolled on the right side of the screen.

## Usage Tips

- **Scrolling**: Use your mouse scroll wheel to navigate the commentary and move history panels on the right.  
- **Previous Move**: A button allows reverting the last move, so you can quickly backtrack.  
- **User Login**: In `User vs User` or `User vs System`, you can set player usernames. The code tracks rating and game history in a rudimentary user profile system.  
- **End of Game**: On checkmate or draw, the system generates a final summary commentary.

## Troubleshooting

- **Engines Not Found**: If Stockfish or Lc0 executables are missing or incorrectly referenced, you may get engine errors. Ensure you’ve downloaded them and updated the paths.
- **Google Generative AI**: If commentary generation fails, confirm your API key is correct and your environment has internet access.
- **Performance**: Running two engines plus commentary generation can be CPU-intensive. Adjust engine analysis depth/time in the code to suit your system.

## Contributing

Contributions, suggestions, or bug reports are welcome!  
1. Fork this repo  
2. Create a new branch (`git checkout -b feature/your_feature`)  
3. Commit changes (`git commit -m "Add your feature"`)  
4. Push to the branch (`git push origin feature/your_feature`)  
5. Open a Pull Request


---

**Happy Chess Gaming & Analyzing!**
