FALLDOWN = """
Analyze this image carefully. Determine if a person has fallen down.

Important classification rules:

- The "falldown" category applies to any person who is lying down, regardless of:
  - the surface (e.g., floor, mattress, bed)
  - the posture (natural or unnatural)
  - the cause (e.g., sleeping, collapsing, lying intentionally)
- This includes:
  - A person lying flat on the ground or other surfaces
  - A person collapsed or sprawled in any lying position
- The "normal" category applies only if the person is:
  - sitting
  - standing
  - kneeling
  - or otherwise upright (not lying down)

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": a brief reason why this classification was made
  (e.g., "person lying on a mattress", "person sitting on sofa")

Example:
{
  "category": "falldown",
  "description": "person lying on a mattress in natural posture"
}
"""
FIRE = """
Analyze this image carefully. Only classify it as "fire"
if there are ACTUAL FLAMES or ACTIVE FIRE visible in the image.

Important classification rules:

- The "fire" category applies ONLY to images with:
 - visible flames or burning fire
 - active combustion or blazing materials
 - actual fire sources (e.g., campfire, house fire, candle flame, burning car, torch flame)
- This includes:
 - Real flames from any source
 - Active burning or combustion
 - Visible fire with smoke and flames
 - Burning vehicles or cars on fire
 - Torches, flamethrowers, or similar devices with visible flames
- The "normal" category applies to:
 - Fire-related objects WITHOUT actual flames (fire trucks, fire extinguishers, fire hydrants)
 - Red colors, sunset glows, red lights, or the sun
 - Fire emojis or cartoon representations of fire
 - Text or words containing "fire" (signs, labels, written text)
 - Any image without visible flames or active fire

Answer in JSON format with BOTH of the following fields:
- "category": either "fire" or "normal"
- "description": a brief reason why this classification was made
(e.g., "visible flames from campfire", "fire truck without flames")

Example:
{
 "category": "fire",
 "description": "visible flames from burning building"
}
"""
