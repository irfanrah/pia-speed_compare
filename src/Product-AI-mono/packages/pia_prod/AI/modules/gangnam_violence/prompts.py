VIOLENCE = """
Watch this short video clip (1–2 seconds) and respond with exactly one JSON object.

[Rules]
- The category must be either 'violence' or 'normal'.
- Classify as violence if any of the following actions are present:
  * Punching
  * Kicking
  * Weapon Threat
  * Weapon Attack
  * Falling/Takedown
  * Pushing/Shoving
  * Brawling/Group Fight
- If none of the above are observed, classify as normal.
- The following cases must always be classified as normal:
  * Affection (hugging, holding hands, light touches)
  * Helping (supporting, assisting)
  * Accidental (unintentional bumping)
  * Playful (non-aggressive playful contact)

[Output Format]u
- Output exactly one JSON object.
- The object must contain only two keys: "category" and "description".
- The description should briefly and objectively describe the scene.

Example (violence):
{"category":"violence","description":"A man in a black jacket punches another man, who stumbles backward."}

Example (normal):
{"category":"normal","description":"Two people are hugging inside an elevator"}
"""
